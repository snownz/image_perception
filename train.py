from src.data import build_data_loader
from src.models import BackbonePerception

from dataclasses import dataclass, field
from typing import List

import torch
import math

import numpy as np

@dataclass
class TrainConfig:
    centering: str = "centering"
    EPOCH_LENGTH: int = 1500
    saveckp_freq: int = 100

@dataclass
class TeacherConfig:
    momentum_teacher: float = 0.92
    final_momentum_teacher: float = 1.0
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 3

@dataclass
class StudentConfigBackboneC:
    in_channels: float = 3

@dataclass
class StudentConfigBackboneT:
    embed_dim: float = 384
    num_heads: float = 12
    depth: float = 6
    num_tokens: float = 4096

@dataclass
class StudentConfigHead:
    out_dim: int = 1024*8
    in_dim: int = 384
    nlayers: int = 3
    hidden_dim: int = 2048

@dataclass
class StudentConfig:
    backbone_c: StudentConfigBackboneC = field( default_factory = StudentConfigBackboneC )
    backbone_t: StudentConfigBackboneT = field( default_factory = StudentConfigBackboneT )
    head: StudentConfigHead = field( default_factory = StudentConfigHead )

@dataclass
class OptimConfig:
    epochs: int = 10
    weight_decay: float = 0.04
    weight_decay_end: float = 0.08
    base_lr: float = 0.0003
    min_lr: float = 0.0001
    warmup_epochs: int = 0
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

@dataclass
class CropsConfig:
    global_crops_scale: List[float] = field(default_factory=lambda: [0.32, 1.0])
    global_crops_size: int = 512

@dataclass
class ComputePrecision:
    grad_scaler: bool = True

@dataclass
class DinoLoss:
    out_dim: int = 1024*8
    student_temp: float = 0.1
    center_momentum: float = 0.9

@dataclass
class iBOTLoss:
    patch_out_dim: int = 1024*8
    student_temp: float = 0.1
    center_momentum: float = 0.9

@dataclass
class VicRegLoss:
    sim_weight: float = 0.1
    var_weight: float = 0.1
    cov_weight: float = 0.1
    eps: float = 1e-6

@dataclass
class Config:
    log_dir: str = "./models/"
    name: str = "backbone_v1_small"
    backbone_arch: str = "small"
    train: TrainConfig = field( default_factory = TrainConfig )
    crops_config: CropsConfig = field( default_factory = CropsConfig )
    student: StudentConfig = field( default_factory = StudentConfig )
    teacher: TeacherConfig = field( default_factory = TeacherConfig )
    optim: OptimConfig = field( default_factory = OptimConfig )
    compute_precision: ComputePrecision = field( default_factory = ComputePrecision )
    dino_loss: DinoLoss = field( default_factory = DinoLoss )
    ibot_loss: iBOTLoss = field( default_factory = iBOTLoss )
    vicreg_loss: VicRegLoss = field( default_factory = VicRegLoss )

cfg = Config()

data_loader = build_data_loader( 
    'dataset', 
    batch_size = 8, 
    global_crops_size = cfg.crops_config.global_crops_size 
)

# torch.autograd.set_detect_anomaly( mode = True, check_nan = True )
device = ( 'cuda' if torch.cuda.is_available() else 'cpu' )

model = BackbonePerception( cfg, device )
model.load()
model.to( device )

from src.utils import CosineScheduler, MetricLogger

def build_optimizer(cfg, params_groups):
    from src.adopt import ADOPT
    return ADOPT( params_groups, lr = cfg.optim.base_lr, betas = ( cfg.optim.adamw_beta1, cfg.optim.adamw_beta2 ), weight_decay = cfg.optim.weight_decay )
    # from torch.optim import AdamW
    # return AdamW( params_groups, lr = cfg.optim.base_lr, betas = ( cfg.optim.adamw_beta1, cfg.optim.adamw_beta2 ), weight_decay = cfg.optim.weight_decay )

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
    momentum = dict(
        base_value = cfg.teacher.momentum_teacher,
        final_value = cfg.teacher.final_momentum_teacher,
        total_iters = cfg.optim.epochs * EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value = cfg.teacher.teacher_temp,
        final_value = cfg.teacher.teacher_temp,
        total_iters = cfg.teacher.warmup_teacher_temp_epochs * EPOCH_LENGTH,
        warmup_iters = cfg.teacher.warmup_teacher_temp_epochs * EPOCH_LENGTH,
        start_warmup_value = cfg.teacher.warmup_teacher_temp,
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        :int( cfg.optim.freeze_last_layer_epochs * EPOCH_LENGTH )
    ] = 0  # mimicking the original schedules

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )

def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):

    for param_group in optimizer.param_groups:

        is_last_layer = param_group.get( "is_last_layer", False )
        lr_multiplier = param_group.get( "lr_multiplier", 1.0 )
        wd_multiplier = param_group.get( "wd_multiplier", 1.0 )
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = ( last_layer_lr if is_last_layer else lr ) * lr_multiplier

optimizer = build_optimizer( cfg, model.get_params_groups() )
(
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    teacher_temp_schedule,
    last_layer_lr_schedule,
) = build_schedulers( cfg )

# set model to train mode and half precision
fp16_scaler = model.fp16_scaler  # for mixed precision training
model.train()

tensorboard_logger = MetricLogger( delimiter = "\t", log_dir = cfg.log_dir + "/" + cfg.name, iteration = model.t_step )
for data in tensorboard_logger.log_every(
    data_loader,
    print_freq = 1,
    header = "Training",
    n_iterations = 20000,
    start_iteration = model.t_step,
):
    
    current_batch_size = data["collated_global_crops"].shape[0] / 2
            
    # schedulers
    lr = lr_schedule[model.t_step]
    wd = wd_schedule[model.t_step]
    mom = momentum_schedule[model.t_step]
    teacher_temp = teacher_temp_schedule[model.t_step]
    last_layer_lr = last_layer_lr_schedule[model.t_step]
    apply_optim_scheduler( optimizer, lr, wd, last_layer_lr )

    # compute losses
    optimizer.zero_grad( set_to_none = True )
    loss_dict = model.forward_backward( data, teacher_temp = teacher_temp, backward = True )

    if math.isnan(sum(loss_dict.values())):  # Detect NaNs in the loss
        print( "NaNs in the loss, skipping the iteration" )
        break
    
    # clip gradients
    if fp16_scaler is not None:
        if cfg.optim.clip_grad:
            fp16_scaler.unscale_( optimizer )
            for v in model.student.values():
                torch.nn.utils.clip_grad_norm_( v.parameters(), max_norm = cfg.optim.clip_grad )
        fp16_scaler.step( optimizer )
        fp16_scaler.update()
    else:
        if cfg.optim.clip_grad:
            for v in model.student.values():
                torch.nn.utils.clip_grad_norm_( v.parameters(), max_norm = cfg.optim.clip_grad )
        optimizer.step()

    # perform teacher EMA update
    model.update_teacher( mom )

    # log losses
    loss_dict_reduced = { k: v for k, v in loss_dict.items() }
    losses_reduced = sum( loss for loss in loss_dict_reduced.values() )

    tensorboard_logger.update( lr = lr )
    tensorboard_logger.update( wd = wd )
    tensorboard_logger.update( mom = mom )
    tensorboard_logger.update( teacher_temp = teacher_temp )
    tensorboard_logger.update( last_layer_lr = last_layer_lr )
    tensorboard_logger.update( current_batch_size = current_batch_size )
    tensorboard_logger.update( total_loss = losses_reduced, **loss_dict_reduced )

    model.t_step += 1

    if model.t_step % cfg.train.saveckp_freq == 0:
        model.save()
