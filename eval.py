from src.data import build_data_loader_train_detection, draw, denormalize_transform
from src.models import ObjectDetectionModel

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
class Detection:
    nc: int = 25
    ch: tuple = (384, 384, 384)
    hd: int = 256  # hidden dim
    nq: int = 300  # num queries
    ndp: int = 4  # num decoder points
    nh: int = 8  # num head
    ndl: int = 6  # num decoder layers
    d_ffn: int = 1024  # dim of feedforward
    dropout: float = 0.0
    act: nn.Module = nn.ReLU()
    eval_idx: int = -1
    # Training args
    learnt_init_query: bool = False

@dataclass
class Config:
    log_dir: str = "./models/"
    name: str = "detection_v2_small"
    backbone_name: str = "encoder_v2"
    compute_precision: ComputePrecision = field( default_factory = ComputePrecision )
    backbone: BackboneConfig = field( default_factory = BackboneConfig )
    hungarian_loss: HungarianLoss = field( default_factory = HungarianLoss )
    detection: Detection = field( default_factory = Detection )

cfg = Config()
device = ( 'cuda' if torch.cuda.is_available() else 'cpu' )

model = ObjectDetectionModel( cfg, device )
model.load()

model.to( device )
model.eval()

import cv2
image = cv2.imread( "test_images/8_p.png" )
image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
torch_image = torch.from_numpy( image ).permute( 2, 0, 1 ).float() / 255.0
torch_image = torch_image.to( device )

model.predict( torch_image, confidence_threshold = 0.5 )


