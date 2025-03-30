from models import BackbonePerception

from dataclasses import dataclass, field
from typing import List

import torch
import math

import numpy as np

@dataclass
class TrainConfig:
    centering: str = "centering"
    EPOCH_LENGTH: int = 1000
    saveckp_freq: int = 100

@dataclass
class TeacherConfigBackboneC:
    in_channels: float = 3

@dataclass
class TeacherConfigBackboneT:
    embed_dim: float = 384
    num_heads: float = 12
    depth: float = 12
    num_tokens: float = 4096

@dataclass
class TeacherConfigHead:
    out_dim: int = 1024*8
    in_dim: int = 384
    nlayers: int = 3
    hidden_dim: int = 2048

@dataclass
class TeacherConfig:
    backbone_c: TeacherConfigBackboneC = field( default_factory = TeacherConfigBackboneC )
    backbone_t: TeacherConfigBackboneT = field( default_factory = TeacherConfigBackboneT )
    head: TeacherConfigHead = field( default_factory = TeacherConfigHead )
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
    weight_decay_end: float = 0.4
    base_lr: float = 0.001
    lr: float = 0
    warmup_epochs: int = 1
    min_lr: float = 1.0e-06
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1
    scaling_rule: str = "sqrt_wrt_1024"
    patch_embed_lr_mult: float = 0.2
    layerwise_decay: float = 0.9
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

@dataclass
class CropsConfig:
    global_crops_scale: List[float] = field(default_factory=lambda: [0.32, 1.0])
    global_crops_size: int = 518

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
    train: TrainConfig = field( default_factory = TrainConfig )
    student: StudentConfig = field( default_factory = StudentConfig )
    teacher: TeacherConfig = field( default_factory = TeacherConfig )
    optim: OptimConfig = field( default_factory = OptimConfig )
    compute_precision: ComputePrecision = field( default_factory = ComputePrecision )
    dino_loss: DinoLoss = field( default_factory = DinoLoss )
    ibot_loss: iBOTLoss = field( default_factory = iBOTLoss )
    vicreg_loss: VicRegLoss = field( default_factory = VicRegLoss )
    backbone_arch = "small"

cfg = Config()
device ='cpu'
model = BackbonePerception( cfg, device )
model.load()
model.to( device )
m = model.build_perception( 'encoder_v1' )
m.save()
