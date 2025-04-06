import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import GradScaler

from src.torch_tools import Module
from src.modules import Encoder, PredictorHead, TransformerDetection, RTDETRDecoder
from src.losses import DINOLoss, iBOTPatchLoss, KoLeoLoss, VicRegLoss, HungarianLossComputation, FocalLoss, normalized_l1_loss, bbox_iou, compute_bbox_classification_loss
from src.detection_training_function import generate_noisy_bboxes, extract_pos_neg_boxes, extract_pos_neg_cls
from src.data import extract_overlapping_crops_and_boxes, merge_feature_maps, nms_torch

import math
from functools import partial
import numpy as np
import time

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False
        # raise ValueError(f"NaN detected in {name}")

class PerceptionModel(Module):

    def __init__(self, config, log_dir, name, device):

        super().__init__()

        self.cfg = config
        log_dir = log_dir
        name = name
        self.log_dir = log_dir + name
        self.dvc = device
        self.backbone = Encoder( **config )

    def forward(self, images, masks=None):
        t = self.backbone( images, masks = masks )
        return t

    def save(self):
        self.save_training( self.log_dir )

    def load(self, chpt=None, eval=True):
        self.load_training( self.log_dir, chpt, eval )

class BackbonePerception(Module):

    def __init__(self, config, device):

        super().__init__()

        self.cfg = config
        log_dir = config.log_dir
        name = config.name
        self.log_dir = log_dir + name
        self.dvc = device

        self.fp16_scaler = GradScaler() if config.compute_precision.grad_scaler else None

        # Teacher Model
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[config.backbone_arch]
        backbone_name = f"dinov2_{backbone_arch}"
        backbone_model = torch.hub.load( repo_or_dir = "facebookresearch/dinov2", model = backbone_name )
        backbone_model.eval()
        backbone_model.cuda()
        self.teacher_forward = partial(
            backbone_model.get_intermediate_layers,
            n = [5, 7, 11],
            reshape = False,
            return_class_token = True,
        )
        self.teacher = nn.ModuleDict({
            'dino_head': PredictorHead( **vars( config.student.head ) ),
        })
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Student Model
        self.backbone_c_student = Backbone( **vars( config.student.backbone_c ) )
        self.backbone_t_student = VisionEncoder( **vars( config.student.backbone_t ) )
        self.dino_head_student = PredictorHead( **vars( config.student.head ) )
        self.student = nn.ModuleDict({
            'backbone_c': self.backbone_c_student,
            'backbone_t': self.backbone_t_student,
            'dino_head': self.dino_head_student,
        })

        # Losses
        self.dino_loss = DINOLoss( **vars( config.dino_loss ) )
        self.ibot_patch_loss = iBOTPatchLoss( **vars( config.ibot_loss ) )
        self.koleo_loss = KoLeoLoss()
        self.vicreg_loss = VicRegLoss( **vars( config.vicreg_loss ) )

    def build_perception(self, name):

        config = {
            'backbone_c': self.cfg.student.backbone_c,
            'backbone_t': self.cfg.student.backbone_t,
        }

        model = PerceptionModel( config, self.cfg.log_dir, name, self.dvc )
        model.backbone_c.load_state_dict( self.backbone_c_student.state_dict() )
        model.backbone_t.load_state_dict( self.backbone_t_student.state_dict() )

        return model

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale( loss ).backward()
        else:
            loss.backward()

    def update_teacher(self, m):
        student_params = list( self.student['dino_head'].parameters() )
        teacher_params = list( self.teacher['dino_head'].parameters() )
        with torch.no_grad():
            torch._foreach_mul_( teacher_params, m )
            torch._foreach_add_( teacher_params, student_params, alpha = 1 - m )

    def forward_backbone_student(self, images,  masks):
        c = self.student['backbone_c']( images )
        t = self.student['backbone_t']( c, masks = masks )
        return t

    @torch.no_grad()
    def forward_backbone_teacher(self, images):

        def downsample_tokens(x, new_token_count):

            bs, num_patches, D = x.shape
            grid_size = int(num_patches**0.5)
            x_grid = x.reshape(bs, grid_size, grid_size, D).permute(0, 3, 1, 2)
            new_grid_size = int(new_token_count**0.5)
            x_down = F.adaptive_avg_pool2d(x_grid, (new_grid_size, new_grid_size))
            x_down = x_down.flatten(2).permute(0, 2, 1)

            return x_down

        s3 = self.teacher_forward( torch.nn.functional.interpolate( images, size = 896 ) )
        return {
            "patch_layers" : [
                {
                    "x_norm_cls_token": s3[0][1], # [bs, D]
                    "x_norm_patch_tokens": s3[0][0] # [bs, 4098, D]
                },
                {
                    "x_norm_cls_token": s3[1][1],
                    "x_norm_patch_tokens": downsample_tokens( s3[1][0], 1024 ) # Reshape to [bs, 1024, D]
                },
                {
                    "x_norm_cls_token": s3[2][1],
                    "x_norm_patch_tokens": downsample_tokens( s3[2][0], 256 ) # Reshape to [bs, 256, D]
                }
            ]
        }

    def forward_backward(self, images, teacher_temp, backward):

        with torch.amp.autocast( enabled = self.cfg.compute_precision.grad_scaler, device_type = "cuda" ):

            # 1) Basic Setup and Input Tensors
            n_global_crops = 2  # We assume exactly 2 global crops.

            # Global crops is a single tensor [B_total, C, H, W].
            # Typically B_total == 2 * batch_size if you stacked them externally.
            global_crops = images["collated_global_crops"].cuda( non_blocking = True )

            masks = [ m.cuda( non_blocking = True ) for m in images["collated_masks"] ]
            mask_indices_list = [ m.cuda( non_blocking = True ) for m in images["mask_indices_list"] ]
            n_masked_patches_tensor = [ m.cuda( non_blocking = True ) for m in images["n_masked_patches"] ]  # (S, ) or something similar
            upperbound = images["upperbound"]                     # list of length S
            masks_weight = [ m.cuda( non_blocking = True ) for m in images["masks_weight"] ]

            # We'll sum up losses here.
            loss_accumulator = 0.0
            loss_dict = {}

            self.dino_loss_weight = 1.0
            self.ibot_loss_weight = 1.0
            self.vicreg_loss_weight = 0.01
            self.koleo_loss_weight = 0.001
            ibot_loss_scale = 1.0 / n_global_crops  # usual iBOT scaling

            # 2) TEACHER FORWARD PASS (Multiple Scales)
            @torch.no_grad()
            def get_teacher_output():
                """
                Runs the teacher backbone on the 2 global crops,
                collects (CLS tokens + masked patch tokens) for each scale,
                passes them through the teacher's (shared) 'dino_head',
                applies centering, and returns them.
                """
                teacher_out = self.forward_backbone_teacher( global_crops )
                # teacher_out["patch_layers"] a list of dicts:
                #   patch_layers[s] = {
                #       "x_norm_cls_token": [2*B, D],
                #       "x_norm_patch_tokens": [2*B, N_s, D],
                #       ...
                #   }

                patch_layers = teacher_out["patch_layers"]  # e.g. s3_t, s4_t, s5_t

                # For centering: we also store the final "softmaxed & centered" distributions
                #   - DINO wants teacher CLS tokens distribution
                #   - iBOT wants teacher masked patch tokens distribution
                teacher_dino_softmaxed_list = []
                teacher_ibot_softmaxed_list = []

                for s, layer_dict in enumerate(patch_layers):

                    # Extract the raw teacher CLS tokens for this scale
                    raw_teacher_cls_tokens = layer_dict["x_norm_cls_token"]  # shape [2*B, D]
                    # Re-chunk and reorder for DINO global crops
                    # chunk(2) => two chunks: [B, D] each, then [::-1] reverses them, then cat => back to [2*B, D].
                    raw_teacher_cls_tokens = raw_teacher_cls_tokens.chunk( n_global_crops )
                    raw_teacher_cls_tokens = torch.cat( [ raw_teacher_cls_tokens[1], raw_teacher_cls_tokens[0] ], dim = 0 )  # [2*B, D]
                    n_cls_tokens = raw_teacher_cls_tokens.shape[0]  # => 2*B

                    # Extract the raw teacher patch tokens for this scale
                    raw_teacher_patch_tokens = layer_dict["x_norm_patch_tokens"]  # shape [2*B, N_s, D]
                    _dim = raw_teacher_patch_tokens.shape[-1]

                    # Prepare a buffer that can hold [CLS tokens + masked patch tokens]
                    #    buffer shape = [u + n_cls_tokens, D]
                    #    where u = upperbound[s], a big enough number for masked patches
                    buffer_teacher = raw_teacher_patch_tokens.new_zeros(
                        upperbound[s] + n_cls_tokens, _dim
                    )

                    # 1) Copy CLS tokens into the front
                    buffer_teacher[:n_cls_tokens].copy_(raw_teacher_cls_tokens)

                    # 2) Gather masked patch tokens
                    # Flatten patch tokens from [2*B, N_s, D] => [2*B*N_s, D],
                    # then index_select the masked positions.
                    n_masked_s = n_masked_patches_tensor[s].item()  # e.g. an integer
                    torch.index_select(
                        raw_teacher_patch_tokens.flatten( 0, 1 ),  # => [2*B*N_s, D]
                        dim = 0,
                        index = mask_indices_list[s],             # => shape [n_masked_s]
                        out = buffer_teacher[n_cls_tokens : n_cls_tokens + n_masked_s],
                    )

                    # 3) Apply the (shared) teacher head (dino_head)
                    tokens_after_head = self.teacher.dino_head( buffer_teacher )
                    # => shape [u + n_cls_tokens, D], but we only care about the first (n_cls_tokens + n_masked_s)

                    # Slice back out:
                    teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]  # [2*B, D]
                    masked_teacher_patch_tokens_after_head = tokens_after_head[n_cls_tokens : n_cls_tokens + n_masked_s]

                    # Now apply centering if configured:
                    if self.cfg.train.centering == "centering":

                        # 3a) DINO CLS tokens => softmax + center
                        teacher_dino_softmaxed = self.dino_loss.softmax_center_teacher(
                            teacher_cls_tokens_after_head, teacher_temp = teacher_temp
                        )
                        self.dino_loss.update_center( teacher_cls_tokens_after_head )

                        # 3b) iBOT masked patch tokens => softmax + center
                        masked_teacher_patch_tokens_uq = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                        teacher_ibot_softmaxed = self.ibot_patch_loss.softmax_center_teacher(
                            masked_teacher_patch_tokens_uq[:, :n_masked_s],
                            teacher_temp=teacher_temp
                        )
                        teacher_ibot_softmaxed = teacher_ibot_softmaxed.squeeze(0)
                        self.ibot_patch_loss.update_center( masked_teacher_patch_tokens_after_head[:n_masked_s] )

                    elif self.cfg.train.centering == "sinkhorn_knopp":

                        # Alternative centering approach
                        teacher_dino_softmaxed = self.dino_loss.sinkhorn_knopp_teacher(
                            teacher_cls_tokens_after_head, teacher_temp = teacher_temp
                        )
                        teacher_ibot_softmaxed = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                            masked_teacher_patch_tokens_after_head,
                            teacher_temp=teacher_temp,
                            n_masked_patches_tensor = n_masked_s,
                        )
                    else:
                        raise NotImplementedError("Unsupported centering method")

                    # Save the results for scale s
                    teacher_dino_softmaxed_list.append( teacher_dino_softmaxed )
                    teacher_ibot_softmaxed_list.append( teacher_ibot_softmaxed )

                # Return all scales
                return ( teacher_dino_softmaxed_list,
                        teacher_ibot_softmaxed_list )

            # Get teacher output
            (
                teacher_dino_softmaxed_list,
                masked_teacher_ibot_softmaxed_list
            ) = get_teacher_output()


            # 3) STUDENT FORWARD PASS (Multiple Scales)
            # Now do a single forward pass of the student backbone for the 2 global crops:
            student_out = self.forward_backbone_student( global_crops, masks )
            student_patch_layers = student_out["patch_layers"]

            # 4) Compute the vicreg_loss
            features_scales = student_out["feature_layers"] # [ f1, f3, f3 ]
            vir_reg_global_loss = 0.0
            for fs in features_scales:
                # fs = [2*bs, c, h, w]
                xf, yf = fs.chunk(2)
                vir_reg_global_loss += self.vicreg_loss( xf, yf )

            # We'll store final (cls, patch) after head for each scale
            student_cls_list = []
            student_cls_after_head_list = []
            student_masked_patch_after_head_list = []

            for s, layer_dict in enumerate( student_patch_layers ):

                # Student raw CLS tokens for scale s
                raw_student_cls_tokens = layer_dict["x_norm_cls_token"]  # [2*B, D]
                n_cls_tokens = raw_student_cls_tokens.shape[0]

                # Student raw patch tokens for scale s
                raw_student_patch_tokens = layer_dict["x_norm_patch_tokens"]  # [2*B, N_s, D]
                _dim = raw_student_patch_tokens.shape[-1]

                # Build a buffer:
                buffer_student = raw_student_patch_tokens.new_zeros( upperbound[s] + n_cls_tokens, _dim )
                # 1) Copy CLS tokens
                buffer_student[:n_cls_tokens].copy_( raw_student_cls_tokens )

                # 2) Copy masked patch tokens
                n_masked_s = n_masked_patches_tensor[s].item()
                # torch.index_select(
                #     raw_student_patch_tokens.flatten(0,1),
                #     dim = 0,
                #     index = mask_indices_list[s],
                #     out = buffer_student[n_cls_tokens : n_cls_tokens + n_masked_s]
                # )
                buffer_student[n_cls_tokens : n_cls_tokens + n_masked_s] = torch.index_select(
                    raw_student_patch_tokens.flatten(0,1),
                    dim = 0,
                    index = mask_indices_list[s]
                )

                # 3) Forward through student's (shared) dino_head
                tokens_after_head = self.student.dino_head( buffer_student )
                student_cls_after_head = tokens_after_head[:n_cls_tokens]
                student_masked_patch_after_head = tokens_after_head[n_cls_tokens : n_cls_tokens + n_masked_s]

                student_cls_after_head_list.append( student_cls_after_head )
                student_masked_patch_after_head_list.append( student_masked_patch_after_head )
                student_cls_list.append( raw_student_cls_tokens )

            # 4) Compute Losses per Scale
            # Sum or average across scales.
            total_ibot_loss = 0.0

            # DINO is *only* on CLS tokens
            # We'll compare (student vs teacher) for each scale,
            # or sometimes only the highest scale. Let's sum them all as an example:
            dino_loss_s = self.dino_loss(
                student_output_list = student_cls_after_head_list,
                teacher_out_softmaxed_centered_list = teacher_dino_softmaxed_list,
            )
            total_dino_loss = dino_loss_s

            # iBOT is on *masked patch tokens*
            for s in range( len( student_masked_patch_after_head_list ) ):
                # student => [n_masked_s, D]
                # teacher => masked_teacher_ibot_softmaxed_list[s] => same shape
                ibot_loss_s = self.ibot_patch_loss.forward_masked(
                    student_masked_patch_after_head_list[s],
                    masked_teacher_ibot_softmaxed_list[s],
                    student_masks_flat = masks[s],
                    n_masked_patches = n_masked_patches_tensor[s],
                    masks_weight = masks_weight[s],
                )
                total_ibot_loss += ibot_loss_s
            total_ibot_loss *= ibot_loss_scale

            # koLeo loss
            koleo_loss = self.koleo_loss_weight * sum(
                sum(
                    self.koleo_loss( p ) for p in student_cls_tokens.chunk(2)
                ) for student_cls_tokens in student_cls_list
            )  # we don't apply koleo loss between cls tokens of a same image

            dino_global_loss = ( total_dino_loss * self.dino_loss_weight ) / len( student_cls_after_head_list )
            ibot_global_loss = ( total_ibot_loss * self.ibot_loss_weight ) / len( student_masked_patch_after_head_list )
            vir_reg_global_loss = ( vir_reg_global_loss * self.vicreg_loss_weight ) / len( features_scales )
            koleo_loss = ( koleo_loss * self.koleo_loss_weight ) / len( student_cls_list )

        loss_accumulator += dino_global_loss + ibot_global_loss + vir_reg_global_loss + koleo_loss
        loss_dict["dino_global_loss"] = dino_global_loss.item()
        loss_dict["ibot_global_loss"] = ibot_global_loss.item()
        loss_dict["koleo_loss"] = koleo_loss.item()
        loss_dict["vir_reg_loss"] = vir_reg_global_loss.item()

        self.backprop_loss( loss_accumulator )

        # self.loss += loss_accumulator
        # if backward:
        #     self.backprop_loss( self.loss )
        #     self.loss = 0.0

        return loss_dict

    def get_params_groups(self):
        all_params_groups = [
            {
                "params": [ p for p in self.backbone_c_student.parameters() if p.requires_grad ],
                "lr_multiplier": 1.0,
                "wd_multiplier": 1.0,
                "is_last_layer": False,
            },
            {
                "params": [ p for p in self.backbone_t_student.parameters() if p.requires_grad ],
                "lr_multiplier": 1.0,
                "wd_multiplier": 1.0,
                "is_last_layer": False,
            },
            {
                "params": [ p for p in self.dino_head_student.parameters() if p.requires_grad ],
                "lr_multiplier": 1.0,
                "wd_multiplier": 1.0,
                "is_last_layer": True,
            },
        ]
        
        total_trainable_params = sum(
            param.numel() for group in all_params_groups for param in group["params"]
        )
        all_names = [ name for name, p in self.backbone_c_student.named_parameters() if p.requires_grad ]
        all_names += [ name for name, p in self.backbone_t_student.named_parameters() if p.requires_grad ]
        all_names += [ name for name, p in self.dino_head_student.named_parameters() if p.requires_grad ]        
        print("Total number of trainable layers:", len( all_names ))
        print("Total number of trainable parameters:", total_trainable_params)
        
        return all_params_groups

    def save(self):
        self.save_training( self.log_dir )

    def load(self, chpt=None, eval=True):
        self.load_training( self.log_dir, chpt, eval )

class ObjectDetectionModel(Module):

    def __init__(self, config, device):

        super().__init__()

        self.cfg = config
        log_dir = config.log_dir
        name = config.name
        self.log_dir = log_dir + name
        self.dvc = device

        self.fp16_scaler = GradScaler() if config.compute_precision.grad_scaler else None

        self.backbone = PerceptionModel( name = config.backbone_name, log_dir = config.log_dir, device = device, config = vars( config.backbone ) )
        self.detection = RTDETRDecoder( **vars( config.detection ) )

        if hasattr( config, "detection_loss" ):
            self.hungarian_loss = HungarianLossComputation( **vars( config.hungarian_loss ) )
            self.focal_loss = FocalLoss() 

            class_weights = np.zeros( config.hungarian_loss.num_classes )
            class_weights[0] = 0.1
            class_weights[1:] = 1.0
            self.ce_loss = nn.CrossEntropyLoss( weight = torch.tensor( class_weights ).float().cuda() )

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale( loss ).backward()
        else:
            loss.backward()

    def forward_backward(self, data, num_noise=10, compute_loss=True, compute_noise=True):
        
        max_num_objects = 30

        with torch.amp.autocast( enabled = self.cfg.compute_precision.grad_scaler, device_type = "cuda" ):

            # Get data
            # start = time.time()
            images, labels, bounding_boxes, polygons, mask_patches = data
            ( masks_pacthes, _, _, _ ) = mask_patches

            images = images.cuda( non_blocking = True )
            labels = labels.cuda( non_blocking = True )
            bounding_boxes = bounding_boxes.cuda( non_blocking = True )
            masks_pacthes = masks_pacthes.cuda( non_blocking = True )

            num_objects = bounding_boxes.shape[1]
            if num_objects > max_num_objects:
                bounding_boxes = bounding_boxes[:, :num_objects, :]
                labels = labels[:, :num_objects]
            elif num_objects < max_num_objects:
                num_missing = max_num_objects - num_objects
                bounding_boxes = F.pad( bounding_boxes, ( 0, 0, 0, num_missing, 0, 0 ), value = 0 )
                labels = F.pad( labels, ( 0, num_missing ), value = self.hungarian_loss.num_classes )

            # polygons = polygons.cuda( non_blocking = True )
            # if bin_objects:
            #     labels = torch.clamp( labels, 0, 1 ).long()
            # print("Data loaded in: ", time.time() - start)
            
            # Generate noisy boxes
            # start = time.time()
            if compute_noise:
                noisy_boxes, noise_classes = generate_noisy_bboxes( 
                    bounding_boxes, 
                    labels,
                    self.cfg.hungarian_loss.num_classes,
                    num_copies = num_noise, 
                    box_noise_scale = 0.1,
                    cls_noise_ratio = 0.7
                ) # [ sequence_predctions_postivive_0, sequence_predctions_negative_0, sequence_predctions_postivive_1, sequence_predctions_negative_1, ... ]
            else:
                noisy_boxes = None
                noise_classes = None
            # print("Noisy boxes generated in: ", time.time() - start)

            # Forward Encoder
            # start = time.time()
            features = self.backbone( images, masks = masks_pacthes ) # Forward the backbone, use mask patches to hide the patches and force the model to learn the rest of the image
            # features = self.backbone( images, masks = None )
            # print("Backbone features generated in: ", time.time() - start)

            # Forward Detection
            # start = time.time()
            ( 
                pred_bboxes, 
                pred_label_scores, 
                pred_proposal_bboxes, 
                pred_proposal_label_scores 
            ) = self.detection( features, noisy_boxes, noise_classes, noise_groups = num_noise*int(compute_noise), num_objects = bounding_boxes.shape[1] * 2*int(compute_noise) )
            # print("Detection features generated in: ", time.time() - start)
            
            # split the noise boxes and real boxes
            # start = time.time()
            if compute_noise:
                noise_size = 2 * ( num_noise * bounding_boxes.shape[1] )
                num_pred_tokens = pred_bboxes.shape[2] - noise_size 
                pred_real_bboxes, pred_noise_bboxes = pred_bboxes.split( [ num_pred_tokens, noise_size ], dim = 2 )
                pred_real_label_scores, pred_noise_label_scores = pred_label_scores.split( [ num_pred_tokens, noise_size ], dim = 2 )
            else:
                pred_real_bboxes = pred_bboxes
                pred_real_label_scores = pred_label_scores

            pred_real_bboxes = torch.cat( [ pred_proposal_bboxes.unsqueeze(0), pred_real_bboxes ] ) # Concat the enc_bboxes as a new layer
            pred_real_label_scores = torch.cat( [ pred_proposal_label_scores.unsqueeze(0), pred_real_label_scores ] ) # Concat the enc_bboxes as a new layer
            # print("Detection features split in: ", time.time() - start)

            # start = time.time()
            # 1. Hungarian Loss on the last Layer using real boxes
            ( 
                ( loss_accumulator, loss_dict ), 
                ( sorted_pred_boxes, sorted_pred_cls_logits, sorted_target_boxes, sorted_target_labels ) 
            ) = self.hungarian_loss( pred_real_bboxes[-1], bounding_boxes, pred_real_label_scores[-1], labels )

            # 2. Hungarian Loss on all the other layers using real boxes
            for i in range( 0, pred_real_bboxes.shape[0] - 1 ):
                ( ( loss_accumulator_, loss_dict_ ), _ ) = self.hungarian_loss( 
                    pred_real_bboxes[i], 
                    bounding_boxes, 
                    pred_real_label_scores[i], 
                    labels, 
                )
                loss_accumulator += loss_accumulator_
                loss_dict.update( { f'{k}_{i}': v for k, v in loss_dict_.items() } )
            # print("Hungarian loss computed in: ", time.time() - start)

            if compute_noise:

                # 3. Loss on the positive noise boxes (last layer)
                # self.hungarian_loss.compute_loss() l1 + giou + cls
                # For positive we should align wit the gt boxes, and filter out the non-object classes (0), and tehm pass all the boxes to the loss function
                # pred_positive_nboxes = [bs, num_noise*N, 4]
                # pred_positive_ncls = [bs, num_noise*N, C]
                # bounding_boxes = [bs, N, 4]
                # labels = [bs, N]
                
                # start = time.time()
                B, N = bounding_boxes.shape[:2]
                num_pos = num_noise * N 
                gt_boxes_rep = bounding_boxes.unsqueeze( 2 ).expand( B, N, num_noise, 4 ).reshape( B, num_pos, 4 )
                gt_labels_rep = labels.unsqueeze(1).expand(B, num_noise, N).reshape( B, num_pos )
                object_mask = gt_labels_rep != 0 

                pred_positive_nboxes, pred_negative_nboxes = extract_pos_neg_boxes( pred_noise_bboxes, bounding_boxes )
                pred_positive_ncls, pred_negative_ncls = extract_pos_neg_cls( pred_noise_label_scores, labels )
                # print("Noise boxes extracted in: ", time.time() - start)
                
                # start = time.time()
                noise_loss_accumulator = 0.0
                gt_scores = bbox_iou( pred_positive_nboxes[-1], gt_boxes_rep )
                loss_noise, loss_dict_noise = self.hungarian_loss.compute_loss(
                    pred_positive_nboxes[-1],   # predictions [B, num_pos, 4]
                    gt_boxes_rep,               # targets [B, num_pos, 4]
                    pred_positive_ncls[-1],     # classification predictions [B, num_pos, C]
                    gt_labels_rep,              # target labels [B, num_pos]
                    gt_scores,
                    mask = object_mask
                )
                noise_loss_accumulator += loss_noise
                loss_dict.update( { f'{k}_noise': v for k, v in loss_dict_noise.items() } )

                # 3.1 Loss on the positive noise boxes (all layers)
                for i in range( 0, pred_real_bboxes.shape[0] - 1 ):
                    gt_scores = bbox_iou( pred_positive_nboxes[i], gt_boxes_rep )
                    loss_noise, loss_dict_noise = self.hungarian_loss.compute_loss(
                        pred_positive_nboxes[i],    # predictions [B, num_pos, 4]
                        gt_boxes_rep,               # targets [B, num_pos, 4]
                        pred_positive_ncls[i],      # classification predictions [B, num_pos, C]
                        gt_labels_rep,              # target labels [B, num_pos]
                        gt_scores,
                        mask = object_mask
                    )
                    noise_loss_accumulator += loss_noise
                    loss_dict.update( { f'{k}_noise_{i}': v for k, v in loss_dict_noise.items() } )
                # print("Noise loss computed in: ", time.time() - start)

                # 4. Loss on the negative noise boxes (last layer)
                # start = time.time()
                non_object_loss = 0.0
                for i in range( 0, pred_negative_ncls.shape[0] ):

                    # Push all the negative boxes to the non-object class
                    l = self.focal_loss( pred_negative_ncls[i].flatten( 0, 1 ), torch.zeros( pred_negative_ncls[i].shape[0:2] ).flatten( 0, 1 ).long().to( images.device ) ).sum(1).mean()
                    non_object_loss += l
                    loss_dict.update( { f'non_object_loss_{i}': l.item() } )

                    # Apply contrastive loss  between the negative and positive boxes
                    pos_boxes_last = pred_positive_nboxes[i]  # [B, num_pos, 4]
                    neg_boxes_last = pred_negative_nboxes[i]  # [B, num_pos, 4]

                    # Minimize the iou between the positive and negative boxes
                    iou = bbox_iou( pos_boxes_last.unsqueeze(1), neg_boxes_last.unsqueeze(2) )
                    contrastive_loss = iou.mean()
                    non_object_loss += contrastive_loss
                    loss_dict.update( { f'contrastive_loss_{i}': contrastive_loss.item() } )
                # print("Non-object loss computed in: ", time.time() - start)

                loss_accumulator = (
                    loss_accumulator
                    + 0.1 * non_object_loss  
                    + 0.1 * noise_loss_accumulator
                )

            if compute_loss:
                # start = time.time()
                self.backprop_loss( loss_accumulator )
                # print("Backprop loss computed in: ", time.time() - start)

            output_dict = {
                'pred_boxes': sorted_pred_boxes.detach().cpu(),
                'pred_cls': sorted_pred_cls_logits.argmax(-1).detach().cpu(),
                'target_boxes': sorted_target_boxes.detach().cpu(),
                'target_labels': sorted_target_labels.detach().cpu(),
                'o_pred_boxes': pred_real_bboxes.detach().cpu()[-1],
                'o_pred_cls': pred_real_label_scores.argmax(-1).detach().cpu()[-1],
            }

            return loss_dict, output_dict

    def predict(self, image, stride_slices=512, confidence_threshold=0.5, iou_threshold=0.5, id_bg=-1):

        with torch.amp.autocast( enabled = True, device_type = "cuda" ):

            with torch.no_grad():
                
                h, w = image.shape[1], image.shape[2]
                crops, c_boxes = extract_overlapping_crops_and_boxes( image, 640, stride = stride_slices )
                crops = torch.stack( crops, dim = 0 ).cuda( non_blocking = True )

                # Forward Encoder
                features = self.backbone( crops )

                # Merge the image back to the original size
                features = [
                    merge_feature_maps( c_boxes, features[0], 640, ( w, h ), 80 ).unsqueeze( 0 ),
                    merge_feature_maps( c_boxes, features[1], 640, ( w, h ), 40 ).unsqueeze( 0 ),
                    merge_feature_maps( c_boxes, features[2], 640, ( w, h ), 20 ).unsqueeze( 0 ),
                ]

                # Forward Detection
                pred = self.detection( features ) # [ bs, objects, 4+classes ]
                boxes = pred[0,:,:4]
                scores = pred[0,:,4:].max(-1)[0]
                labels = pred[0,:,4:].argmax(-1)
                detectors_index = torch.arange( 0, boxes.shape[0], device = boxes.device )

                # Filter non-object classes
                object_mask = labels != id_bg
                boxes = boxes[ object_mask ]
                scores = scores[ object_mask ]
                labels = labels[ object_mask ]
                detectors_index = detectors_index[ object_mask ]

                # Filter low confidence scores
                score_mask = scores > confidence_threshold
                boxes = boxes[ score_mask ]
                scores = scores[ score_mask ]
                labels = labels[ score_mask ]
                detectors_index = detectors_index[ score_mask ]

                # NMS
                indices = nms_torch( boxes, scores, iou_threshold )
                boxes = boxes[ indices ].cpu()
                scores = scores[ indices ].cpu()
                labels = labels[ indices ].cpu()
                detectors_index = detectors_index[ indices ].cpu()

                return boxes, scores, labels, detectors_index

    def get_params_groups(self):

        full_trainable = ['dec_score_head.', 'enc_score_head.', 'denoising_class_embed.', 'tgt_embed.', '.15.', 'mask_token']
        def is_full_trainable(param_name):
            return any(sub in param_name for sub in full_trainable)

        # For self.detection
        detection_named_params = list(self.detection.named_parameters())
        detection_full_group = {
            "params": [ p for name, p in detection_named_params if p.requires_grad and is_full_trainable( name ) ],
            "lr_multiplier": 1.0,
            "wd_multiplier": 1.0,
            "is_last_layer": False,
        }
        detection_other_group = {
            "params": [ p for name, p in detection_named_params if p.requires_grad and not is_full_trainable( name ) ],
            "lr_multiplier": 0.01,
            "wd_multiplier": 0.01,
            "is_last_layer": False,
        }

        # For self.backbone
        backbone_named_params = list( self.backbone.named_parameters() )
        backbone_full_group = {
            "params": [ p for name, p in backbone_named_params if p.requires_grad and is_full_trainable( name ) ],
            "lr_multiplier": 1.0,
            "wd_multiplier": 1.0,
            "is_last_layer": False,
        }
        backbone_other_group = {
            "params": [ p for name, p in backbone_named_params if p.requires_grad and not is_full_trainable( name ) ],
            "lr_multiplier": 0.01,
            "wd_multiplier": 0.01,
            "is_last_layer": False,
        }

        all_params_groups = [
            detection_full_group,
            detection_other_group,
            backbone_full_group,
            backbone_other_group
        ]
        total_trainable_params = sum(
            param.numel() for group in all_params_groups for param in group["params"]
        )
        all_names = [ name for name, p in self.detection.named_parameters() if p.requires_grad ]
        all_names += [ name for name, p in self.backbone.named_parameters() if p.requires_grad ]
        
        print("Total number of trainable layers:", len( all_names ))
        print("Total number of trainable parameters:", total_trainable_params)
        return all_params_groups

    def save(self):
        self.backbone.save()
        self.save_training( self.log_dir )

    def train(self):
        self.backbone.train()
        self.detection.train()
    
    def eval(self):
        self.backbone.eval()
        self.detection.eval()

    def load(self, chpt=None, eval=True):
        self.load_training( self.log_dir, chpt, eval )

