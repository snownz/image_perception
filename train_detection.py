from src.data import build_data_loader_train_detection, draw
from src.models import ObjectDetectionModel
from src.adopt import ADOPT

from dataclasses import dataclass, field

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
    EPOCH_LENGTH: int = 100
    saveckp_freq: int = 100

@dataclass
class OptimConfig:
    epochs: int = 100
    weight_decay: float = 0.0005
    weight_decay_end: float = 0.0005
    base_lr: float = 0.000357
    min_lr: float = 0.0000357
    base_momentum: float = 0.937
    warmup_epochs: int = 3
    clip_grad: float = 3.0
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

@dataclass
class ConfigBackbone:
    in_channels: float = 3
    embed_dim: float = 512
    num_heads: float = 8
    depth: float = 4
    num_tokens: float = 4096
    model = ''

@dataclass
class BackboneConfig:
    in_channels: float = 3
    embed_dim: float = 384
    num_heads: float = 8
    depth: float = 4
    num_tokens: float = 4096
    model: str = "linear"

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
    crop_size: int = 640

@dataclass
class Detection:
    nc: int = 24
    ch: tuple = ( 384, 384, 384 )
    hd: int = 256  # hidden dim
    nq: int = 300  # num queries
    ndp: int = 4  # num decoder points
    nh: int = 8  # num head
    ndl: int = 6  # num decoder layers
    d_ffn: int = 1024  # dim of feedforward
    dropout: float = 0.0
    act: nn.Module = nn.ReLU()
    eval_idx: int = -1
    learnt_init_query: bool = False

@dataclass
class Config:
    log_dir: str = "./models/"
    name: str = "detection_v5_small"
    backbone_name: str = "encoder_v5"
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

data_loader = build_data_loader_train_detection( 
    'dataset_detection', 
    batch_size = 16, 
    max_objects = cfg.detection.nq, 
    max_poly_points = cfg.dataset.max_poly_points,
    # max_poly_points = 5,
    crop_size = cfg.dataset.crop_size,
    mode = 'seg' 
)

model = ObjectDetectionModel( cfg, device )
# model.load()
model.to( device )

from src.utils import CosineScheduler, MetricLogger

def build_optimizer(cfg, params_groups):
    from torch.optim import AdamW
    return ADOPT( params_groups, lr = cfg.optim.base_lr, betas = ( cfg.optim.adamw_beta1, cfg.optim.adamw_beta2 ), weight_decay = cfg.optim.weight_decay, eps = 1e-8 )
    # return AdamW( params_groups, lr = cfg.optim.base_lr, betas = ( cfg.optim.adamw_beta1, cfg.optim.adamw_beta2 ), weight_decay = cfg.optim.weight_decay, momentum = cfg.optim.base_momentum, eps = 1e-8 )

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
model.t_step = 0

tensorboard_logger = MetricLogger( delimiter = "\t", log_dir = cfg.log_dir + "/" + cfg.name, iteration = model.t_step )
for data in tensorboard_logger.log_every(
    data_loader,
    print_freq = 1,
    header = "Training",
    n_iterations = model.t_step + 200000,
    start_iteration = model.t_step,
):

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

    model.t_step += 1

    if model.t_step % cfg.train.saveckp_freq == 0:
        model.save()
