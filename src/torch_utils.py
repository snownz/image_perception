import torch
import torch.nn as nn
import copy
import math
import numpy as np
from torch.nn.init import uniform_

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * ( k - 1 ) + 1 if isinstance( k, int ) else [ d * ( x - 1 ) + 1 for x in k ]
    if p is None:
        p = k // 2 if isinstance( k, int ) else [ x // 2 for x in k ]
    return p

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp( min = 0, max = 1 )
    x1 = x.clamp( min = eps )
    x2 = ( 1 - x ).clamp( min = eps )
    return torch.log( x1 / x2 )

def get_clones(module, n):
    return nn.ModuleList( [ copy.deepcopy( module ) for _ in range( n ) ] )

def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value."""
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init


def linear_init(module):
    """Initialize the weights and biases of a linear module."""
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)