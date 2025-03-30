from data import build_data_loader_train_detection, draw, denormalize_transform
from models import ObjectDetectionModel

from dataclasses import dataclass, field
from typing import List

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import torch
import math

import numpy as np
from torch import nn
import torch.nn.functional as F

def resize_image(image_tensor, size):
    return F.interpolate( image_tensor.unsqueeze(0), size = ( size, size ),
                          mode = 'bilinear', align_corners = False ).squeeze(0)

@dataclass
class TrainConfig:
    EPOCH_LENGTH: int = 5000
    saveckp_freq: int = 100

@dataclass
class OptimConfig:
    epochs: int = 100
    weight_decay: float = 1e-6
    weight_decay_end: float = 1e-4
    base_lr: float = 1e-4
    min_lr: float = 1e-5
    warmup_epochs: int = 0
    clip_grad: float = 3.0
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

@dataclass
class ConfigBackboneC:
    in_channels: float = 3
    out_channels: float = 512
    use_resnet: bool = True

@dataclass
class ConfigBackboneT:
    embed_dim: float = 512
    num_heads: float = 8
    depth: float = 4
    num_tokens: float = 4096

@dataclass
class BackboneConfig:
    backbone_c: ConfigBackboneC = field( default_factory = ConfigBackboneC )
    backbone_t: ConfigBackboneT = field( default_factory = ConfigBackboneT )

@dataclass
class ComputePrecision:
    grad_scaler: bool = True

@dataclass
class HungarianLoss:
    lambda_bbox: int = 1.0
    lambda_cls: float = 1.0
    image_width: float = 512
    image_height: float = 512
    num_classes: int = 25

@dataclass
class DatasetConfig:
    max_poly_points: int = 128

@dataclass
class Detection:
    feature_dim: int = 512
    num_heads: int = 4
    num_layers: int = 3
    num_detections: int = 60
    num_classes: int = 25
    num_bins: int = 100

@dataclass
class Config:
    log_dir: str = "./models/"
    name: str = "detection_v0_small"
    backbone_name: str = "encoder_v3"
    train: TrainConfig = field( default_factory = TrainConfig )
    optim: OptimConfig = field( default_factory = OptimConfig )
    compute_precision: ComputePrecision = field( default_factory = ComputePrecision )
    backbone: BackboneConfig = field( default_factory = BackboneConfig )
    hungarian_loss: HungarianLoss = field( default_factory = HungarianLoss )
    detection: Detection = field( default_factory = Detection )
    dataset: DatasetConfig = field( default_factory = DatasetConfig )

cfg = Config()
# torch.autograd.set_detect_anomaly( mode = True, check_nan = True )
device = ( 'cuda' if torch.cuda.is_available() else 'cpu' )

data_loader = build_data_loader_train_detection( 'dataset_detection', 
                                                  batch_size = 16, 
                                                  max_objects = cfg.detection.num_detections, 
                                                  max_poly_points = cfg.dataset.max_poly_points,
                                                  crop_size = 512,
                                                  mode = 'seg' )

model = ObjectDetectionModel( cfg, device )
model.load()
model.to( device )

from utils import CosineScheduler, MetricLogger

def build_optimizer(cfg, params_groups):
    from torch.optim import AdamW
    return AdamW( params_groups, lr = cfg.optim.base_lr, betas = ( cfg.optim.adamw_beta1, cfg.optim.adamw_beta2 ), weight_decay = cfg.optim.weight_decay )

def build_schedulers(cfg):

    EPOCH_LENGTH = cfg.train.EPOCH_LENGTH
    lr = dict(
        base_value = cfg.optim.base_lr,
        final_value = cfg.optim.min_lr,
        total_iters = cfg.optim.epochs * EPOCH_LENGTH,
        warmup_iters = cfg.optim.warmup_epochs * EPOCH_LENGTH,
        start_warmup_value = 0,
    )
    wd = dict(
        base_value = cfg.optim.weight_decay,
        final_value = cfg.optim.weight_decay_end,
        total_iters = cfg.optim.epochs * EPOCH_LENGTH,
    )
    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)

    return (
        lr_schedule,
        wd_schedule,
    )

def apply_optim_scheduler(optimizer, lr, wd):

    for param_group in optimizer.param_groups:

        lr_multiplier = param_group.get( "lr_multiplier", 1.0 )
        wd_multiplier = param_group.get( "wd_multiplier", 1.0 )
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] =  lr * lr_multiplier

optimizer = build_optimizer( cfg, model.get_params_groups() )
(
    lr_schedule,
    wd_schedule,
) = build_schedulers( cfg )

# set model to train mode and half precision
fp16_scaler = model.fp16_scaler  # for mixed precision training
model.train()

tensorboard_logger = MetricLogger( delimiter = "\t", log_dir = cfg.log_dir + "/" + cfg.name, iteration = model.t_step )
for data in tensorboard_logger.log_every(
    data_loader,
    print_freq = 1,
    header = "Training",
    n_iterations = model.t_step + 20000,
    start_iteration = model.t_step,
):
    pass

    # schedulers
    lr = lr_schedule[model.t_step]
    wd = wd_schedule[model.t_step]
    apply_optim_scheduler( optimizer, lr, wd )

    # compute losses
    optimizer.zero_grad( set_to_none = True )
    loss_dict, output_dict = model.forward_backward( data )

    if math.isnan(sum(loss_dict.values())):  # Detect NaNs in the loss
        print( "NaNs in the loss, skipping the iteration" )
        break
    
    # clip gradients
    if fp16_scaler is not None:
        if cfg.optim.clip_grad:
            fp16_scaler.unscale_( optimizer )
            torch.nn.utils.clip_grad_norm_( model.detection.parameters(), max_norm = cfg.optim.clip_grad )
        fp16_scaler.step( optimizer )
        fp16_scaler.update()
    else:
        if cfg.optim.clip_grad:
            torch.nn.utils.clip_grad_norm_( model.detection.parameters(), max_norm = cfg.optim.clip_grad )
        optimizer.step()

    # log losses
    loss_dict_reduced = { k: v for k, v in loss_dict.items() }
    losses_reduced = sum( loss for loss in loss_dict_reduced.values() )

    tensorboard_logger.update( lr = lr )
    tensorboard_logger.update( wd = wd )
    tensorboard_logger.update( total_loss = losses_reduced, **loss_dict_reduced )

    accuracy = accuracy_score( output_dict["target_labels"], output_dict["pred_cls"] )
    precision = precision_score( output_dict["target_labels"], output_dict["pred_cls"], average = "micro" )
    f1 = f1_score( output_dict["target_labels"], output_dict["pred_cls"], average = "micro" )
    recall = recall_score( output_dict["target_labels"], output_dict["pred_cls"], average = "micro" )
    tensorboard_logger.update( metric_accuracy = accuracy, metric_precision = precision, metric_f1 = f1, metric_recall = recall )

    if model.t_step % 20 == 0:

        data[2] = output_dict["o_pred_boxes"]
        drawn_img = draw( data, grid_cols = 4 )
        tensorboard_logger.update_image( prediction = drawn_img )

        # four_random_indices = np.random.randint( 0, data[0].shape[0], 4 )
        # images = data[0]
        # masks = output_dict["proposal_regions"]
        # images_list = []
        # for i in range( 4 ):
        #     img = []
        #     img_idx = four_random_indices[i]
        #     for j in range( 3 ):
        #         mask = masks[j][img_idx]
        #         mask = ( mask > 0.5 ).float()
        #         im_size = 512
        #         mask_size = mask.shape[1]
        #         random_mask_color = torch.zeros_like( images[img_idx] )
        #         random_mask_color += torch.rand( 3 )[:,None,None]
        #         mask = resize_image( mask[None], im_size )
        #         img.append(  ( mask * denormalize_transform( images[img_idx] ) ) + ( ( 1 - mask ) *  random_mask_color) )
        #     images_list.append( torch.cat( img, dim = 2 ) )
        # grid = torch.cat( images_list, dim = 1 )
        # tensorboard_logger.update_image( region_prediction = grid )

    model.t_step += 1

    if model.t_step % cfg.train.saveckp_freq == 0:
        model.save()
