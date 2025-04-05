import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import ResNet50_Weights
from torch.nn.init import constant_, xavier_uniform_, uniform_

from src.torch_utils import autopad, inverse_sigmoid, get_clones, bias_init_with_prob, linear_init
from src.layers import Attention, MemEffAttention, MLP, SwiGLUFFNFused, DropPath, LayerScale, Conv, LightConv, DWConv, SequentialGraph, RepConv, Concat
from src.layers import drop_add_residual_stochastic_depth

import numpy as np
import time

from functools import partial
import math

class AttentionBlock(nn.Module):
    
    def __init__( self, dim, num_heads, mlp_ratio=4.0, 
                  qkv_bias=False, proj_bias=True, ffn_bias=True,
                  drop: float=0.0, attn_drop=0.0, init_values=None, drop_path=0.0,
                  act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_class=Attention, ffn_layer=MLP):
        
        super().__init__()
        
        # Norm Layer + Attention + DropPath + LayerScale -> attn_residual_func
        self.norm1 = norm_layer( dim )
        self.attn = attn_class( dim,
            num_heads = num_heads,
            qkv_bias = qkv_bias,
            proj_bias = proj_bias,
            attn_drop = attn_drop,
            proj_drop = drop,
        )
        self.ls1 = LayerScale( dim, init_values = init_values ) if init_values else nn.Identity()
        self.drop_path1 = DropPath( drop_path ) if drop_path > 0.0 else nn.Identity()
        
        # Norm Layer + MLP + DropPath + LayerScale -> ffn_residual_func
        self.norm2 = norm_layer( dim )
        mlp_hidden_dim = int( dim * mlp_ratio )
        self.mlp = ffn_layer(
            input_dim = dim,
            hidden_dim = mlp_hidden_dim,
            act = act_layer,
            output_dim = dim,
            drop = drop,
            bias = ffn_bias,
            num_layers = 2,
        )
        self.ls2 = LayerScale( dim, init_values = init_values ) if init_values else nn.Identity()
        self.drop_path2 = DropPath( drop_path ) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x, memory=None, mask=None):
        
        def attn_residual_func(x, memory=None, mask=None):
            n = self.norm1( x ) # Normalize the input tensor
            if memory is not None: 
                a_out, a, penalty = self.attn.forward_cross( n, memory, mask )
            else:
                a_out, a, penalty = self.attn( n ) # Apply attention to the normalized input tensor
            ls = self.ls1( a_out ) # Apply LayerScale to the attention output
            return ls, a, penalty

        def ffn_residual_func(x):
            n = self.norm2( x ) # Normalize the input tensor
            m = self.mlp( n ) # Apply the MLP to the normalized input tensor
            ls = self.ls2( m ) # Apply LayerScale to the MLP output
            return ls, None, None

        if self.training and self.sample_drop_ratio > 0.1:
            
            # the overhead is compensated only for a drop path rate larger than 0.1
            x, a, penalty = drop_add_residual_stochastic_depth(
                (x, memory, mask) if memory is not None else x,
                residual_func = attn_residual_func,
                sample_drop_ratio = self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func = ffn_residual_func,
                sample_drop_ratio = self.sample_drop_ratio,
            )[0]

        elif self.training and self.sample_drop_ratio > 0.0:
            
            a_out, a, penalty = attn_residual_func( x, memory ) # Residual function for the attention block [bs, n, d]
            x = x + self.drop_path1( a_out ) # Drop path for the attention block and add to the input tensor

            a_out = ffn_residual_func( x )[0] # Residual function for the MLP block
            x = x + self.drop_path2( a_out ) # Drop path for the MLP block and add to the input tensor
        
        else:

            a_out, a, penalty = attn_residual_func( x, memory, mask ) # Residual function for the attention block
            x = x + a_out # Add the residual to the input tensor
            x = x + ffn_residual_func( x )[0] # Residual function for the MLP block and add to the input tensor
        
        return x, a, penalty

class Transformer(nn.Module):
    
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.15,
        drop_path_uniform=False,
        init_values=3.0,  # for layerscale: None or 0 => no layerscale
        act_layer=nn.GELU,
        block_fn=AttentionBlock,
        ffn_layer="mlp",
        attn_class=MemEffAttention,
        layers_return=None,
    ):
        super().__init__()
        norm_layer = partial( nn.LayerNorm, eps = 1e-6 )

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads

        if drop_path_uniform is True:
            dpr = [ drop_path_rate ] * depth
        else:
            dpr = [ x.item() for x in torch.linspace( 0, drop_path_rate, depth ) ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer = MLP
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim = embed_dim,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                proj_bias = proj_bias,
                ffn_bias = ffn_bias,
                drop_path = dpr[i],
                norm_layer = norm_layer,
                act_layer = act_layer,
                ffn_layer = ffn_layer,
                init_values = init_values,
                attn_class=attn_class
            )
            for i in range(depth)
        ]
        
        self.chunked_blocks = False
        self.blocks = SequentialGraph( 
            *blocks_list,
            layers_return = layers_return,
            graph = {
                i: { 'input': None, 'additional': [ 'mask', 'memory' ], 'return': 0 } for i in range( len( blocks_list ) )
            }
        )

        self.norm = norm_layer( embed_dim )

    def forward_features(self, x, memory=None, masks=None):

        if isinstance( x, list ):
            return self.forward_features_list( x, masks )
        
        x = self.blocks( x, { 'mask': masks, 'memory': memory } )
        if isinstance( x, list ):
            x = torch.stack( x, dim = 2 ) # [ B N L D ]
        x_norm = self.norm( x )

        return {
            "x_norm_cls_token": x_norm[:,0],
            "x_norm_patch_tokens": x_norm[:,1:],
        }

    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret

class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        
        super().__init__()

        self.ma = nn.MultiheadAttention( c1, num_heads, dropout = dropout, batch_first = True )

        self.fc1 = nn.Linear( c1, cm )
        self.fc2 = nn.Linear( cm, c1 )

        self.norm1 = nn.LayerNorm( c1 )
        self.norm2 = nn.LayerNorm( c1 )
        self.dropout = nn.Dropout( dropout, inplace = True )
        self.dropout1 = nn.Dropout( dropout, inplace = True  )
        self.dropout2 = nn.Dropout( dropout, inplace = True  )

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        
        q = k = self.with_pos_embed( src, pos )
        src2 = self.ma( q, k, value = src, attn_mask = src_mask, key_padding_mask = src_key_padding_mask )[0]
        src = src + self.dropout1( src2 )
        src = self.norm1( src )
        src2 = self.fc2( self.dropout( self.act( self.fc1( src ) ) ) )
        src = src + self.dropout2( src2 )
        return self.norm2( src )

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        
        src2 = self.norm1( src )
        q = k = self.with_pos_embed( src2, pos )
        src2 = self.ma( q, k, value = src2, attn_mask = src_mask, key_padding_mask = src_key_padding_mask )[0]
        src = src + self.dropout1( src2 )
        src2 = self.norm2( src )
        src2 = self.fc2( self.dropout( self.act( self.fc1( src2 ) ) ) )
        return src + self.dropout2( src2 )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        
        if self.normalize_before:
            return self.forward_pre( src, src_mask, src_key_padding_mask, pos )
        return self.forward_post( src, src_mask, src_key_padding_mask, pos )

class AIFI(TransformerEncoderLayer):
    
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
    
        super().__init__( c1, cm, num_heads, dropout, act, normalize_before )

    def forward(self, x):
        
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding( w, h, c )
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward( x.flatten( 2 ).permute( 0, 2, 1 ), pos = pos_embed.to( device = x.device, dtype = x.dtype ) )
        return x.permute( 0, 2, 1 ).view( [ -1, c, h, w ] ).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        
        grid_w = torch.arange( w, dtype = torch.float32 )
        grid_h = torch.arange( h, dtype = torch.float32 )
        grid_w, grid_h = torch.meshgrid( grid_w, grid_h, indexing = "ij" )
        pos_dim = embed_dim // 4
        omega = torch.arange( pos_dim, dtype = torch.float32 ) / pos_dim
        omega = 1.0 / ( temperature**omega )

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat( [ torch.sin( out_w ), torch.cos( out_w ), torch.sin( out_h ), torch.cos( out_h ) ], 1 )[None]

class SimpleBackbone(nn.Module):

    def __init__(self, out_channels=384):
    
        super(SimpleBackbone, self).__init__()

        # input = 896
        self.stage1 = nn.Sequential(
            nn.Conv2d( in_channels, 64, kernel_size = 3, stride = 2, padding = 1 ),  # 512->256 - 1/2 scale
            nn.BatchNorm2d( 64 ),
            nn.ReLU( inplace = True )
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d( 64, 128, kernel_size = 3, stride = 2, padding = 1 ),  # 256->128 - 1/4 scale
            nn.BatchNorm2d( 128 ),
            nn.ReLU( inplace = True )
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d( 128, 256, kernel_size = 3, stride = 2, padding = 1 ),  # 128->64 - 1/8 scale
            nn.BatchNorm2d( 256 ),
            nn.ReLU(inplace=True)
        ) # 64x64
        self.def_stage3 = nn.Sequential(
            DeformableConv2d( 256, out_channels, kernel_size = 3, padding = 1, stride = 1 ),
            nn.BatchNorm2d( out_channels ),
            nn.ReLU( inplace = True )
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d( 256, 512, kernel_size = 3, stride = 2, padding = 1 ),  # 64->32 - 1/16 scale
            nn.BatchNorm2d( 512 ),
            nn.ReLU( inplace = True )
        ) # 32x32
        self.def_stage4 = nn.Sequential( 
            DeformableConv2d( 512, out_channels, kernel_size = 3, padding = 1, stride = 1 ),
            nn.BatchNorm2d( out_channels ),
            nn.ReLU( inplace = True )
        )

        self.stage5 = nn.Sequential(
            nn.Conv2d( 512, 512, kernel_size = 3, stride = 2, padding = 1 ),  # 32->16 - 1/32 scale
            nn.BatchNorm2d( 512 ),
            nn.ReLU( inplace = True )
        ) # 16x16
        self.def_stage5 = nn.Sequential( 
            DeformableConv2d( 512, out_channels, kernel_size = 3, padding = 1, stride = 1 ),
            nn.BatchNorm2d( out_channels ),
            nn.ReLU( inplace = True )
        )

    def forward(self, x):
        
        features = self.stage2( self.stage1( x ) )
        h = self.stage3( features )
        s3 = self.def_stage3( h ) # 1/8 scale
        h = self.stage4( h )
        s4 = self.def_stage4( h ) # 1/16 scale
        h = self.stage5( h )
        s5 = self.def_stage5( h ) # 1/32 scale

        return [ s3, s4, s5 ]

class FrozenResNetBackbone(nn.Module):
    
    def __init__(self, out_channels=384):
    
        super(FrozenResNetBackbone, self).__init__()
        backbone = resnet50( weights = ResNet50_Weights.DEFAULT )
    
        # Freeze all parameters in the backbone.
        for param in backbone.parameters():
            param.requires_grad = False
        
        # Extract intermediate features using IntermediateLayerGetter.
        # We use:
        #   layer2 output → feat64
        #   layer3 output → feat32
        #   layer4 output → feat16
        self.body = IntermediateLayerGetter( backbone, return_layers = {'layer2': 'feat64', 'layer3': 'feat32', 'layer4': 'feat16'} )
        for param in self.body.parameters():
            param.requires_grad = False
        # layer3 (feat64) originally has 512 channels.
        self.reduce64 = DeformableConv2d( 512, out_channels, kernel_size = 3, padding = 1, stride = 1 )
        # layer4 (feat32) originally has 1024 channels.
        self.reduce32 = DeformableConv2d( 1024, out_channels, kernel_size = 3, padding = 1, stride = 1 )        
        # layer5 (feat16) originally has 2048 channels.
        self.reduce16 = DeformableConv2d( 2048, out_channels, kernel_size = 3, padding = 1, stride = 1 )
        
    def forward(self, x):
        # Forward pass through the backbone is done with no gradient.
        with torch.no_grad():
            features = self.body(x)  # Returns a dict: {'feat64': ..., 'feat32': ...}
        
        # Apply the deformable reduction layers.
        feat64 = self.reduce64( features['feat64'] )   # Expected shape: [B, out_channels, 64, 64]
        feat32 = self.reduce32( features['feat32'] )   # Expected shape: [B, out_channels, 32, 32]
        feat16 = self.reduce16( features['feat16'] )    # Expected shape: [B, out_channels, 16, 16]
        
        return [ feat64, feat32, feat16 ]

# Cross-scale Context Fusion Module (CCFM)
class CCFM(nn.Module):

    def __init__(self, embed_dim, num_heads, attn_layer=MemEffAttention):
    
        super().__init__()
        self.attn = attn_layer( embed_dim, num_heads )
        self.linear1 = nn.Linear( embed_dim, embed_dim*4 )
        self.dropout = nn.Dropout( 0.1 )
        self.linear2 = nn.Linear( embed_dim*4, embed_dim )
        self.norm1 = nn.LayerNorm( embed_dim )
        self.norm2 = nn.LayerNorm( embed_dim )

    def forward(self, tgt, memory, mask=None):

        # Cross-attention (tgt attends memory)
        attn_output, _, _ = self.attn.forward_cross( tgt, memory, mask )
        tgt2 = self.norm1( tgt + attn_output )

        # Feed-forward
        ff_output = self.linear2( self.dropout( F.relu( self.linear1( tgt2 ) ) ) )
        tgt3 = self.norm2( tgt2 + ff_output )

        return tgt3

# Context Extractor
class CE(nn.Module):

    def __init__(self, embed_dim, num_heads, num_tokens=383):
    
        super().__init__()
        self.encode = MemEffAttention( embed_dim, num_heads )
        self.linear1 = nn.Linear( embed_dim, embed_dim*4 )
        self.dropout = nn.Dropout( 0.1 )
        self.linear2 = nn.Linear( embed_dim*4, embed_dim )
        self.norm1 = nn.LayerNorm( embed_dim )

        self.decode = MemEffAttention( embed_dim, num_heads )
        self.linear3 = nn.Linear( embed_dim, embed_dim*4 )
        self.dropout2 = nn.Dropout( 0.1 )
        self.linear4 = nn.Linear( embed_dim*4, embed_dim )
        self.norm2 = nn.LayerNorm( embed_dim )
        self.memory_tokens = nn.Parameter( torch.randn( 1, num_tokens, embed_dim ) )

    def forward(self, x):

        # Encode
        x, _, _ = self.encode( x )
        x = self.norm1( x + self.linear2( self.dropout( F.relu( self.linear1( x ) ) ) ) )

        # Decode
        x, _, _ = self.decode.forward_cross( self.memory_tokens.expand( x.shape[0], -1, -1 ), x )
        x = self.norm2( x + self.linear4( self.dropout2( F.relu( self.linear3( x ) ) ) ) )

        return x

class HGStem(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels):

        super(HGStem, self).__init__()

        self.stem1 = Conv( in_channels, mid_channels, 3, 2, act = nn.ReLU(inplace=True) )
        self.stem2a = Conv( mid_channels, mid_channels // 2, 2, 1, 0, act = nn.ReLU(inplace=True) )
        self.stem2b = Conv( mid_channels // 2, mid_channels, 2, 1, 0, act = nn.ReLU(inplace=True) )
        self.stem3 = Conv( mid_channels * 2, mid_channels, 3, 2, act = nn.ReLU(inplace=True) )
        self.stem4 = Conv( mid_channels, out_channels, 1, 1, act = nn.ReLU(inplace=True) )
        self.pool = nn.MaxPool2d( kernel_size = 2, stride = 1, padding = 0, ceil_mode = True )

    def forward(self, x):

        x = self.stem1( x )
        x = F.pad( x, [ 0, 1, 0, 1 ] )
        x2 = self.stem2a( x )
        x2 = F.pad( x2, [ 0, 1, 0, 1 ] )
        x2 = self.stem2b( x2 )
        x1 = self.pool( x )
        x = torch.cat( [ x1, x2 ], dim = 1 )
        x = self.stem3( x )
        x = self.stem4( x )

        return x

class HGBlock(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels, k, lightconv=False, num_layers=1, shortcut=False, act=nn.ReLU(inplace=True)):

        super(HGBlock, self).__init__()

        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList( block( in_channels if i == 0 else mid_channels, mid_channels, k = k, act = act ) for i in range( num_layers ) )
        self.sc = Conv( in_channels + num_layers * mid_channels, out_channels // 2, 1, 1, act = act )
        self.ec = Conv( out_channels // 2, out_channels, 1, 1, act = act )
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):

        y = [x]
        y.extend( m( y[-1] ) for m in self.m )
        y = self.ec( self.sc( torch.cat( y, 1 ) ) )
        
        return y + x if self.add else y

class RepC3(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_layers=1, e=0.5):

        super(RepC3, self).__init__()
        
        c_ = int( out_channels * e )  # hidden channels
        self.cv1 = Conv( in_channels, c_, 1, 1 )
        self.cv2 = Conv( in_channels, c_, 1, 1 )
        self.m = nn.Sequential( *[ RepConv( c_, c_ ) for _ in range( num_layers ) ] )
        self.cv3 = Conv( c_, out_channels, 1, 1 ) if c_ != out_channels else nn.Identity()

    def forward(self, x):
        
        return self.cv3( self.m( self.cv1( x ) ) + self.cv2( x ) )

class AIFI_2(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, depth=4, num_tokens=1024):
        
        super().__init__()

        self.embed_dim = embed_dim
        self.interpolate_offset = 0.1
        self.interpolate_antialias = True

        # CLS token
        self.cls_token = nn.Parameter( torch.randn( 1, 1, embed_dim ) )
        self.pos_embed = nn.Parameter( torch.randn( 1, num_tokens + 1, embed_dim ) )
        self.mask_token = nn.Parameter( torch.zeros( 1, embed_dim ) )

        # AIFI module all scales (assuming 3 scales: S3, S4, S5)
        self.model = Transformer( embed_dim, depth, num_heads )

        self.init_weights()
    
    def init_weights(self):
        trunc_normal_( self.pos_embed, std = 0.02 ) # For positional embeddings
        nn.init.normal_( self.cls_token, std = 1e-6 ) # For the class token

    def interpolate_pos_encoding(self, w0, h0, x):

        # Store the original data type of x to cast the final output back.
        previous_dtype = x.dtype

        # Determine the number of patch tokens in x (subtracting the class token).
        npatch = x.shape[1] - 1

        # N is the number of patch position embeddings originally stored (excluding the class token).
        N = self.pos_embed.shape[1] - 1

        # If the number of patches and resolution are unchanged, return the original positional embeddings.
        if npatch == N:
            return self.pos_embed

        # Convert the positional embeddings to float for interpolation purposes.
        pos_embed = self.pos_embed.float()

        # Separate the class token embedding (first token) from the patch tokens.
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        # Get the embedding dimension.
        dim = x.shape[-1]

        # w0 # Number of patches along width.
        # h0 # Number of patches along height.

        # Calculate M, the original number of patches per dimension.
        # Since the original patch embeddings are arranged in a square grid, M is the square root of N.
        M = int( math.sqrt( N ) )
        assert N == M * M, "The number of original patches must form a square grid (M x M)."

        # Prepare keyword arguments for interpolation.
        kwargs = {}
        if self.interpolate_offset:
            # When an offset is provided, adjust the scale factor.
            # This slight adjustment helps avoid floating point errors during interpolation.
            # The scale factor is computed as (new_size + offset) / original_size for both dimensions.
            sx = float( w0 + self.interpolate_offset ) / M
            sy = float( h0 + self.interpolate_offset ) / M
            kwargs["scale_factor"] = ( sx, sy )
        else:
            # Without an offset, directly specify the target size.
            kwargs["size"] = ( w0, h0 )

        # Reshape the patch positional embeddings from (1, N, dim) to (1, M, M, dim)
        # and then permute to (1, dim, M, M) as required by the interpolation function.
        patch_pos_embed = patch_pos_embed.reshape( 1, M, M, dim ).permute( 0, 3, 1, 2 )

        # Interpolate the grid of patch embeddings to the new grid size (w0 x h0)
        # using bicubic interpolation. Optionally use antialiasing if specified.
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            mode = "bicubic",
            antialias = self.interpolate_antialias,
            **kwargs,
        )

        # Verify that the interpolated grid dimensions match the expected (w0, h0).
        assert ( w0, h0 ) == patch_pos_embed.shape[-2:], "Interpolated grid size does not match expected dimensions."

        # Permute back from (1, dim, w0, h0) to (1, w0, h0, dim) and flatten the grid to (1, w0*h0, dim)
        patch_pos_embed = patch_pos_embed.permute( 0, 2, 3, 1 ).view( 1, -1, dim )

        # Concatenate the class token embedding (expanded to match dimensions) with the interpolated patch embeddings,
        # and convert the result back to the original data type.
        return torch.cat( ( class_pos_embed.unsqueeze(0), patch_pos_embed ), dim = 1 ).to( previous_dtype )

    def prepare_tokens_with_masks(self, x, masks=None):
        
        _, _, w, h = x.shape
        flatten_patches = x.permute( 0, 2, 3, 1 ).flatten( 1, 2 ) # [ B HW D ]
        if masks is not None:
            x = torch.where( masks.unsqueeze(-1), self.mask_token.to( x.dtype ).unsqueeze(0), flatten_patches ) # mask tokens
        else:
            x = flatten_patches
        x = torch.cat( ( self.cls_token.expand( x.shape[0], -1, -1 ), x ), dim = 1 ) # [ B HW+1 D ] Add class token
        x = x + self.interpolate_pos_encoding( w, h, x ) # [ B HW+1 D ] Add positional encoding

        return x

    def forward(self, x, masks=None):

        b, c, w, h = x.shape
        x = self.model( self.prepare_tokens_with_masks( x, masks ) )
        x = x['x_norm_patch_tokens'] # [ B N D ]
        x = x.view( b, w, h, c ).permute( 0, 3, 1, 2 ) # [ B D W H ]
        return x

class LinearEncoder(nn.Module):
    
    def __init__(self, in_channels, embed_dim, depth, num_heads, num_tokens):
        
        super(LinearEncoder, self).__init__()
        
        # input = 640x640
        self.model = SequentialGraph(

            HGStem( in_channels, 32, 64 ),                                                      # 0: Stem
            HGBlock( 64, 64, 128, 3, num_layers = 6 ),                                          # 1

            DWConv( 128, 128, 3, 2 ),                                                           # 2
            HGBlock( 128, 128, 512, 3, num_layers = 6 ),                                        # 3
            HGBlock( 512, 128, 512, 3, num_layers = 6, shortcut = True ),                       # 4

            DWConv( 512, 512, 3, 2 ),                                                           # 5
            HGBlock( 512, 256, 1024, 5, num_layers = 6, lightconv = True ),                     # 6
            HGBlock( 1024, 256, 1024, 5, num_layers = 6, lightconv = True, shortcut = True ),   # 7
            HGBlock( 1024, 256, 1024, 5, num_layers = 6, lightconv = True, shortcut = True ),   # 8
            HGBlock( 1024, 256, 1024, 5, num_layers = 6, lightconv = True, shortcut = True ),   # 9
            HGBlock( 1024, 256, 1024, 5, num_layers = 6, lightconv = True, shortcut = True ),   # 10

            DWConv( 1024, 1024, 3, 2 ),                                                         # 11
            HGBlock( 1024, 512, 2048, 5, num_layers = 6, lightconv = True ),                    # 12
            HGBlock( 2048, 512, 2048, 5, num_layers = 6, lightconv = True, shortcut = True ),   # 13

            Conv( 2048, embed_dim, k = 1, s = 1, act = nn.Identity() ),                         # 14
            AIFI( embed_dim, num_heads = 12, normalize_before = False ),                         # 15
            Conv( embed_dim, embed_dim, k = 1, s = 1 ),                                         # 16

            nn.Upsample( scale_factor = 2.0, mode = 'nearest' ),                                # 17
            Conv( 1024, embed_dim, k = 1, s = 1, act = nn.Identity(), use_mask = True ),        # 18
            Concat(),                                                                           # 19
            RepC3( 768, embed_dim, num_layers = 3, e = 1.0 ),                                   # 20
            Conv( embed_dim, embed_dim, k = 1, s = 1 ),                                         # 21
            
            nn.Upsample( scale_factor = 2.0, mode = 'nearest' ),                                # 22
            Conv( 512, embed_dim, k = 1, s = 1, act = nn.Identity(), use_mask = True ),         # 23
            Concat(),                                                                           # 24
            RepC3( 768, embed_dim, num_layers = 3, e = 1.0 ),                                   # 25
            
            Conv( embed_dim, embed_dim, k = 3, s = 2, p = 1 ),                                  # 26
            Concat(),                                                                           # 27
            RepC3( 768, embed_dim, num_layers = 3, e = 1.0 ),                                   # 28
            
            Conv( embed_dim, embed_dim, k = 3, s = 2, p = 1 ),                                  # 29
            Concat(),                                                                           # 30
            RepC3( 768, embed_dim, num_layers = 3, e = 1.0 ),                                   # 31
            
            layers_return = [ 25, 28, 31 ],
            graph = {
                18: { 'input': 10 },
                19: { 'input': [ 17, 18 ] },
                23: { 'input': 4 },
                24: { 'input': [ 22, 23 ] },
                27: { 'input': [ 26, 21 ] },
                30: { 'input': [ 29, 16 ] },
            },
            # graph = {
            #     15: { 'input': None, 'additional': [ 'masks' ] },
            #     18: { 'input': 10, 'additional': [ 'masks' ] },
            #     19: { 'input': [ 17, 18 ] },
            #     23: { 'input': 4, 'additional': [ 'masks' ] },
            #     24: { 'input': [ 22, 23 ] },
            #     27: { 'input': [ 26, 21 ] },
            #     30: { 'input': [ 29, 16 ] },
            # },
            # grads = [15]
        )

        self.load_from_pt()

    def forward(self, x, masks=None):
        return self.model( x, { 'masks': masks } ) # [ B, C, H, W ] -> [ B, C, H/8, W/8 ], [ B, C, H/16, W/16 ], [ B, C, H/32, W/32 ]

    def load_from_pt(self, local='models/pre_trained/rtdetr-c.pth'):

        l_state_dict = torch.load( local, map_location = 'cpu', weights_only = False )()
        c_state_dict = self.state_dict()
        l_names = { '.'.join( x.split('.')[2:] ): x  for x in l_state_dict.keys() if int( x.split('.')[2] ) < 32 }
        c_names = [ '.'.join( x.split('.')[2:] ) for x in c_state_dict.keys() ]
        load_dict = {}
        for l_name in l_names:
            if l_name in c_names:
                c_shape = c_state_dict['model.layers.'+l_name].shape
                l_shape = l_state_dict[l_names[l_name]].shape
                if c_shape != l_shape:
                    raise( f"Warning: {l_name} shape mismatch: {c_shape} != {l_shape}" )
                else:
                    load_dict['model.layers.'+l_name] = l_state_dict[l_names[l_name]]
            else:
                print( f"Warning: {l_name} not found in current model." )
        self.load_state_dict( load_dict, strict = False )
        print( f"Loaded {len(load_dict)} layers from pre-trained model." )

class SeparatedScaleEncoder(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, depth=4, num_tokens=1024):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.interpolate_offset = 0.1
        self.interpolate_antialias = True

        # CLS token
        self.cls_token = nn.Parameter( torch.randn( 1, 1, embed_dim ) )
        self.pos_embed = nn.Parameter( torch.randn( 1, num_tokens + 1, embed_dim ) )
        self.hier_embed = nn.Parameter( torch.randn( 2, 1, 1, embed_dim ) )
        self.mask_token = nn.Parameter( torch.zeros( 1, embed_dim ) )

        # AIFI module all scales (assuming 3 scales: S3, S4, S5)
        self.aifi = Transformer( embed_dim, depth, num_heads, skip_first_residual = False )

        # CE module to extract context
        self.ce3 = CE( embed_dim, num_heads, 128 )
        self.ce4 = CE( embed_dim, num_heads, 128 )
        self.ce5 = CE( embed_dim, num_heads, 128 )

        # CCFM modules to fuse cross-scale context
        self.ccfm_s3 = CCFM( embed_dim, num_heads )
        self.ccfm_s4 = CCFM( embed_dim, num_heads )
        self.ccfm_s5 = CCFM( embed_dim, num_heads )

        self.head = nn.Identity()
        self.init_weights()
    
    def init_weights(self):
        trunc_normal_( self.pos_embed, std = 0.02 ) # For positional embeddings
        nn.init.normal_( self.cls_token, std = 1e-6 ) # For the class token

    def interpolate_pos_encoding(self, w0, h0, x):

        # Store the original data type of x to cast the final output back.
        previous_dtype = x.dtype

        # Determine the number of patch tokens in x (subtracting the class token).
        npatch = x.shape[1] - 1

        # N is the number of patch position embeddings originally stored (excluding the class token).
        N = self.pos_embed.shape[1] - 1

        # If the number of patches and resolution are unchanged, return the original positional embeddings.
        if npatch == N:
            return self.pos_embed

        # Convert the positional embeddings to float for interpolation purposes.
        pos_embed = self.pos_embed.float()

        # Separate the class token embedding (first token) from the patch tokens.
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        # Get the embedding dimension.
        dim = x.shape[-1]

        # w0 # Number of patches along width.
        # h0 # Number of patches along height.

        # Calculate M, the original number of patches per dimension.
        # Since the original patch embeddings are arranged in a square grid, M is the square root of N.
        M = int( math.sqrt( N ) )
        assert N == M * M, "The number of original patches must form a square grid (M x M)."

        # Prepare keyword arguments for interpolation.
        kwargs = {}
        if self.interpolate_offset:
            # When an offset is provided, adjust the scale factor.
            # This slight adjustment helps avoid floating point errors during interpolation.
            # The scale factor is computed as (new_size + offset) / original_size for both dimensions.
            sx = float( w0 + self.interpolate_offset ) / M
            sy = float( h0 + self.interpolate_offset ) / M
            kwargs["scale_factor"] = ( sx, sy )
        else:
            # Without an offset, directly specify the target size.
            kwargs["size"] = ( w0, h0 )

        # Reshape the patch positional embeddings from (1, N, dim) to (1, M, M, dim)
        # and then permute to (1, dim, M, M) as required by the interpolation function.
        patch_pos_embed = patch_pos_embed.reshape( 1, M, M, dim ).permute( 0, 3, 1, 2 )

        # Interpolate the grid of patch embeddings to the new grid size (w0 x h0)
        # using bicubic interpolation. Optionally use antialiasing if specified.
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            mode = "bicubic",
            antialias = self.interpolate_antialias,
            **kwargs,
        )

        # Verify that the interpolated grid dimensions match the expected (w0, h0).
        assert ( w0, h0 ) == patch_pos_embed.shape[-2:], "Interpolated grid size does not match expected dimensions."

        # Permute back from (1, dim, w0, h0) to (1, w0, h0, dim) and flatten the grid to (1, w0*h0, dim)
        patch_pos_embed = patch_pos_embed.permute( 0, 2, 3, 1 ).view( 1, -1, dim )

        # Concatenate the class token embedding (expanded to match dimensions) with the interpolated patch embeddings,
        # and convert the result back to the original data type.
        return torch.cat( ( class_pos_embed.unsqueeze(0), patch_pos_embed ), dim = 1 ).to( previous_dtype )

    def prepare_tokens_with_masks(self, x, masks=None):
        
        B, nc, w, h = x.shape
        flatten_patches = x.permute( 0, 2, 3, 1 ).flatten( 1, 2 ) # [ B HW D ]
        if masks is not None:
            x = torch.where( masks.unsqueeze(-1), self.mask_token.to( x.dtype ).unsqueeze(0), flatten_patches ) # mask tokens
        else:
            # identity pass
            x = self.head( flatten_patches )

        x = torch.cat( ( self.cls_token.expand( x.shape[0], -1, -1 ), x ), dim = 1 ) # [ B HW+1 D ] Add class token
        x = x + self.interpolate_pos_encoding( w, h, x ) # [ B HW+1 D ] Add positional encoding

        return x

    def forward(self, features, masks=None):

        # features: list of feature maps [S3, S4, S5], each [B, C, H, W]
        s3, s4, s5 = features
        
        # Apply AIFI (self-attention within scale)
        s3_aifi = self.aifi( self.prepare_tokens_with_masks( s3, masks[0] ), is_training = self.training )
        s4_aifi = self.aifi( self.prepare_tokens_with_masks( s4, masks[1] ), is_training = self.training )
        s5_aifi = self.aifi( self.prepare_tokens_with_masks( s5, masks[2] ), is_training = self.training )

        # Fuse cross-scale features using CCFM
        # For S3 (fuses with S4 and S5)        
        s3_t = self.ccfm_s3( 
            s3_aifi['x_norm_patch_tokens'], 
            torch.cat( 
                [ 
                    self.ce3( 
                        self.prepare_tokens_with_masks( F.interpolate( s4, size = s3.shape[-2:] ), None ).add_( self.hier_embed[0] )
                    )[:,1:], 
                    self.ce3( 
                        self.prepare_tokens_with_masks( F.interpolate( s5, size = s3.shape[-2:] ), None ).add_( self.hier_embed[1] )
                    )[:,1:] 
                ], 
            dim = 1 ) 
        ) # First scale 1/8, for small objects context

        # For S4 (fuses with S3 and S5)        
        s4_t = self.ccfm_s4( 
            s4_aifi['x_norm_patch_tokens'], 
            torch.cat( 
                [ 
                    self.ce4( 
                        self.prepare_tokens_with_masks( F.interpolate( s3, size = s4.shape[-2:] ), None ).add_( self.hier_embed[0] )
                    )[:,1:], 
                    self.ce4( 
                        self.prepare_tokens_with_masks( F.interpolate( s5, size = s4.shape[-2:] ), None ).add_( self.hier_embed[1] )
                    )[:,1:] 
                ], 
            dim = 1 ) 
        ) # Second scale 1/16, for medium objects context

        # For S5 (fuses with S3 and S4)        
        s5_t = self.ccfm_s5( 
            s5_aifi['x_norm_patch_tokens'], 
            torch.cat( 
                [ 
                    self.ce5( 
                        self.prepare_tokens_with_masks( F.interpolate( s3, size = s5.shape[-2:] ), None ).add_( self.hier_embed[0] )
                    )[:,1:], 
                    self.ce5( 
                        self.prepare_tokens_with_masks( F.interpolate( s4, size = s5.shape[-2:] ), None ).add_( self.hier_embed[1] )
                    )[:,1:] 
                ], 
            dim = 1 ) 
        ) # Third scale 1/32, for large objects context

        return{
            "feature_layers": [ s3, s4, s5 ],
            "patch_layers": [ 
                {
                    "x_norm_cls_token": s3_aifi['x_norm_cls_token'],
                    "x_norm_patch_tokens": s3_t,
                },
                {
                    "x_norm_cls_token": s4_aifi['x_norm_cls_token'],
                    "x_norm_patch_tokens": s4_t,
                },
                {
                    "x_norm_cls_token": s5_aifi['x_norm_cls_token'],
                    "x_norm_patch_tokens": s5_t,
                }
             ]
        }

class Encoder(nn.Module):

    def __init__(self, in_channels, embed_dim, num_heads, depth, num_tokens, model):
        
        super(Encoder, self).__init__()
        
        if model == "resnet50+sep-scale":
            backbone = FrozenResNetBackbone( out_channels = embed_dim )
            encoder = SeparatedScaleEncoder( embed_dim = embed_dim, num_heads = num_heads, depth = depth, num_tokens = num_tokens )
            self.net = nn.Sequential(
                backbone,
                encoder
            )
        elif model == "Simple+sep-scale":
            backbone = SimpleBackbone( in_channels = in_channels, out_channels = embed_dim )
            encoder = SeparatedScaleEncoder( embed_dim = embed_dim, num_heads = num_heads, depth = depth, num_tokens = num_tokens )
            self.net = nn.Sequential(
                backbone,
                encoder
            )
        elif model == "linear":
            self.net = LinearEncoder( in_channels = in_channels, embed_dim = embed_dim, depth = depth, num_heads = num_heads, num_tokens = num_tokens )
        else:
            raise ValueError( f"Unknown model type: {model}" )
    
    def forward(self, x, masks=None):
        return self.net( x, masks )

class PredictorHead(nn.Module):

    def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    
        if nlayers == 1:
            return nn.Linear( in_dim, bottleneck_dim, bias = bias )
        
        else:
            layers = [ nn.Linear( in_dim, hidden_dim, bias = bias ) ]
            if use_bn:
                layers.append( nn.BatchNorm1d( hidden_dim ) )
            layers.append( nn.GELU() )
            for _ in range( nlayers - 2 ):
                layers.append( nn.Linear( hidden_dim, hidden_dim, bias = bias ) )
                if use_bn:
                    layers.append( nn.BatchNorm1d( hidden_dim ) )
                layers.append( nn.GELU() )
            layers.append( nn.Linear( hidden_dim, bottleneck_dim, bias = bias ) )
            
            return nn.Sequential( *layers )

    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, mlp_bias=True):

        super().__init__()
        nlayers = max( nlayers, 1 )
        
        self.mlp = _build_mlp( nlayers, in_dim, bottleneck_dim, hidden_dim = hidden_dim, use_bn = use_bn, bias = mlp_bias ) # stack of Layer+Norm+GELU
        self.apply( self._init_weights )
        self.last_layer = weight_norm( nn.Linear( bottleneck_dim, out_dim, bias = False ) )
        self.last_layer.weight.data.fill_( 1 )

    def _init_weights(self, m):
        if isinstance( m, nn.Linear ):
            trunc_normal_( m.weight, std = 0.02 )
            if isinstance( m, nn.Linear ) and m.bias is not None:
                nn.init.constant_( m.bias, 0 )

    def forward(self, x):

        x = self.mlp( x )
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12

        x = nn.functional.normalize( x, dim = -1, p = 2, eps = eps )
        x = self.last_layer( x )

        return x

class DetectionHead(nn.Module):
    
    def __init__(self, feature_dim, num_classes, num_bins=100):
    
        super( DetectionHead, self ).__init__()
    
        self.num_classes = num_classes
        self.num_bins = num_bins

        self.bbox_head = nn.Sequential(
            nn.Linear( feature_dim, feature_dim ),
            nn.ReLU(),
            nn.Dropout( 0.1 ),
            nn.Linear( feature_dim, feature_dim ),
            nn.ReLU(),
            nn.Dropout( 0.1 ),
            nn.Linear( feature_dim, 4 )
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear( feature_dim, feature_dim ),
            nn.ReLU(),
            nn.Dropout( 0.1 ),
            nn.Linear( feature_dim, feature_dim ),
            nn.ReLU(),
            nn.Dropout( 0.1 ),
            nn.Linear( feature_dim, num_classes )
        )

    def forward(self, features):

        class_logits = self.cls_head( features )  # Shape: [batch_size, seq_len, num_classes]
        boxe_values = self.bbox_head( features ).sigmoid()  # Shape: [batch_size, seq_len, 4]
        return boxe_values, class_logits

class TransformerDetection(nn.Module):
    
    def __init__(self, feature_dim, num_heads, num_layers, num_detections, num_bins, num_classes):
    
        super( TransformerDetection, self ).__init__()

        self.num_detections = num_detections
    
        self.tokens_objects = nn.Parameter( torch.randn( 1, num_detections, feature_dim ), requires_grad = True )  # Queries for small objects

        self.box_projection = nn.Linear( 4, feature_dim )  # Projection noise boxes to small query objects
        self.cls_embedding = nn.Linear( 2, feature_dim )  # Class embedding

        self.pos_embed = nn.Parameter( torch.randn( 1, 4096, feature_dim ), requires_grad = True )  # Positional embeddings
        self.cls_token = nn.Parameter( torch.randn( 1, 1, feature_dim ), requires_grad = True )  # Class token

        self.decoder = Transformer( embed_dim = feature_dim, depth = num_layers, num_heads = num_heads, attn_class = Attention )

        self.detection_head = DetectionHead( feature_dim, num_classes, num_bins )  # Detection head
        self.detection_head_bin = DetectionHead( feature_dim, 2, num_bins )  # Detection head

        self.interpolate_offset = 0.1
        self.interpolate_antialias = True

    def init_weights(self):

        # Now we do a small uniform init so the output is typically ~[-1..1].
        nn.init.uniform_( self.box_projection.weight, a = -0.2, b = 0.2 )
        nn.init.uniform_( self.box_projection.bias, a = -0.2, b = 0.2 )
        nn.init.uniform_( self.cls_embedding.weight, a = -0.5, b = 0.5 )
        nn.init.uniform_( self.cls_embedding.bias, a = -0.5, b = 0.5 )

    def interpolate_pos_encoding(self, w0, h0, x):

        # Store the original data type of x to cast the final output back.
        previous_dtype = x.dtype

        # Determine the number of patch tokens in x (subtracting the class token).
        npatch = x.shape[1]

        # N is the number of patch position embeddings originally stored (excluding the class token).
        N = self.pos_embed.shape[1]

        # If the number of patches and resolution are unchanged, return the original positional embeddings.
        if npatch == N:
            return self.pos_embed

        # Convert the positional embeddings to float for interpolation purposes.
        pos_embed = self.pos_embed.float()

        # Get the embedding dimension.
        dim = pos_embed.shape[-1]

        # Calculate M, the original number of patches per dimension.
        # Since the original patch embeddings are arranged in a square grid, M is the square root of N.
        M = int( math.sqrt( N ) )
        assert N == M * M, "The number of original patches must form a square grid (M x M)."

        # Prepare keyword arguments for interpolation.
        kwargs = {}
        # When an offset is provided, adjust the scale factor.
        # This slight adjustment helps avoid floating point errors during interpolation.
        # The scale factor is computed as (new_size + offset) / original_size for both dimensions.
        sx = float( w0 + self.interpolate_offset ) / M
        sy = float( h0 + self.interpolate_offset ) / M
        kwargs["scale_factor"] = ( sx, sy )

        # Reshape the patch positional embeddings from (1, N, dim) to (1, M, M, dim)
        # and then permute to (1, dim, M, M) as required by the interpolation function.
        pos_embed = pos_embed.reshape( 1, M, M, dim ).permute( 0, 3, 1, 2 )

        # Interpolate the grid of patch embeddings to the new grid size (w0 x h0)
        # using bicubic interpolation. Optionally use antialiasing if specified.
        pos_embed = nn.functional.interpolate(
            pos_embed,
            mode = "bicubic",
            antialias = self.interpolate_antialias,
            **kwargs,
        )

        # Verify that the interpolated grid dimensions match the expected (w0, h0).
        assert ( w0, h0 ) == pos_embed.shape[-2:], "Interpolated grid size does not match expected dimensions."

        # Permute back from (1, dim, w0, h0) to (1, w0, h0, dim) and flatten the grid to (1, w0*h0, dim)
        pos_embed = pos_embed.permute( 0, 2, 3, 1 ).view( 1, -1, dim )

        # Concatenate the class token embedding (expanded to match dimensions) with the interpolated patch embeddings,
        # and convert the result back to the original data type.
        return pos_embed.to( previous_dtype )

    def prepare_memory_tokens(self, x):
        bs, _, w, h = x.shape
        flatten_patches = x.permute( 0, 2, 3, 1 ).flatten( 1, 2 ) # [ B HW D ]
        pos = self.interpolate_pos_encoding( w, h, flatten_patches ) # [ B HW D ] Positional encoding
        x = flatten_patches + pos.expand( bs, -1, -1 )
        return x
    
    def prepare_query_tokens(self, x):
        bs = x.shape[0]
        x = torch.cat( [ self.cls_token.expand( bs, -1, -1 ), x ], dim = 1 ) # [ B HW+1 D ] Add class token
        return x   

    def forward(self, encoder_outputs, noisy_boxes=None, noise_groups=0, num_objects=0):

        @torch.no_grad()
        def build_cross_attn_mask(T, L, M, G, device='cpu'):
            """
            
            T: Number of memory tokens (patch tokens) in the encoder output.
            L: Number of learnable queries (class tokens).
            M: Number of noisy groups.
            G: Number of noisy queries per group.

            Build a boolean mask of shape [N, T + N] for cross-attention, where:
            N = L + M * G  (total number of decoder queries)

            Columns [0..T-1] are the memory tokens (patch tokens):
            - All queries can attend these => mask=False.

            Columns [T..T+N-1] are the query tokens themselves:
            - We partition the row index i and column index j into:
                * learnable block or
                * one of M noisy groups.
            - If both i and j belong to the same block, we allow attention => False
            - Otherwise, mask out => True

            Returns:
            attn_mask: BoolTensor [N, T + N], 
                        where True = "blocked", False = "allowed"
            """

            # 1) Total number of queries
            N = L + M * G

            # 2) Initialize all to True => "blocked"
            attn_mask = torch.ones( ( N, T + N ), dtype = torch.bool, device = device )

            # 3) STEP: let all queries attend memory columns [0..T-1]
            attn_mask[:, :T] = False  # => unmask => allowed

            # Helper to identify (block_type, group_index)
            # for a given query index q in [0..N-1].
            # Rows i => which query is attending
            # Columns j => which query is being attended to (in the [T..T+N-1] region).
            def query_block(q):
                """
                Return ("learnable", -1) if q < L
                Otherwise ("noisy", group_idx) with group_idx in [0..M-1].
                """
                if q < L:
                    return ("learnable", -1)
                else:
                    # It's a noisy query
                    noisy_index = q - L  # how far into the noisy section
                    group_idx = noisy_index // G  # which group the query belongs to
                    return ( "noisy", group_idx )

            # 4) Now handle query–query columns => [T..T+N-1]
            for i in range(N):  # row: which query is attending
                block_i, group_i = query_block(i)
                for j_query in range(N):  # which query is in the column
                    block_j, group_j = query_block( j_query )
                    col = T + j_query  # actual column index in [T..T+N-1]. Skip the first T columns
                    if block_i == block_j:
                        # Both queries in same "type"
                        if block_i == "learnable":
                            # learnable => all attend each other
                            attn_mask[i, col] = False
                        else:
                            # "noisy" => only attend if same group
                            if group_i == group_j:
                                attn_mask[i, col] = False
                        # If block types differ, remain True => blocked
                    # else remain True => blocked
            
            # Convrt to float
            attn_mask = ( attn_mask.float() ) * -1e20
            return attn_mask
        
        bs = encoder_outputs[0].shape[0]
        num_queries = 1 + self.num_detections
        
        # Prepare the KV cahce from the encoder
        small_objects_memory = self.prepare_memory_tokens( encoder_outputs[0] )
        medium_objects_memory = self.prepare_memory_tokens( encoder_outputs[1] )
        large_objects_memory = self.prepare_memory_tokens( encoder_outputs[2] )
        objects_memory = torch.cat( [ small_objects_memory, medium_objects_memory, large_objects_memory ], dim = 1 ) # [bs, num_patches, feature_dim]

        if noisy_boxes is not None:
            noise_query = self.box_projection( noisy_boxes[:,:,:4] ) + self.cls_embedding( noisy_boxes[:,:,4:] ) # [bs, num_detections, feature_dim + 64]
            queries_objects = torch.cat( [ self.tokens_objects.expand( bs, -1, -1 ), noise_query ], dim = 1 ) # [bs, num_detections, feature_dim + 64]
            mask = build_cross_attn_mask( objects_memory.shape[1], num_queries, noise_groups, num_objects, device = small_objects_memory.device )
        else:
            queries_objects = self.tokens_objects.expand( bs, -1, -1 )
            mask = None

        def detect(queries_objects, memory, mask, detection_head_bin):
            objects_decoder_outputs = self.decoder(
                x = self.prepare_query_tokens( queries_objects ),
                memory = memory,
                masks = mask
            )
            bboxes, class_logits = detection_head_bin( objects_decoder_outputs['x_norm_patch_tokens'] )
            return bboxes, class_logits

        # Detection Head
        bboxes, class_logits = detect( queries_objects, objects_memory, mask, self.detection_head_bin ) # Detect small objects
        
        # Iteratively refine all objects
        noise_bboxes = [ bboxes[:,( num_queries - 1 ):] ]
        noise_class_logits = [ class_logits[:,( num_queries - 1 ):] ]
        for _ in range( 3 ):

            bboxes, logits = detect( 
                self.box_projection( bboxes ) + self.cls_embedding( F.softmax( class_logits, dim = -1 ) ),
                objects_memory, 
                mask, 
                self.detection_head_bin 
            )
            noise_bboxes.append( bboxes[:,( num_queries - 1 ):] )
            noise_class_logits.append( logits[:,( num_queries - 1 ):] )

        if noisy_boxes is None:            
            return bboxes, class_logits

        bboxes = bboxes[:,:( num_queries - 1 )]
        class_logits = class_logits[:,:( num_queries - 1 )]
              
        return ( bboxes, class_logits ), ( noise_bboxes, noise_class_logits )

class MSDeformAttn(nn.Module):
    """
    Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels # number of feature levels
        self.n_heads = n_heads # number of attention heads
        self.n_points = n_points # number of sampling points per attention head ( 4 for boxes )

        self.sampling_offsets = nn.Linear( d_model, n_heads * n_levels * n_points * 2 )
        self.attention_weights = nn.Linear( d_model, n_heads * n_levels * n_points )
        self.value_proj = nn.Linear( d_model, d_model )
        self.output_proj = nn.Linear( d_model, d_model )

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_( self.sampling_offsets.weight.data, 0.0 )
        thetas = torch.arange( self.n_heads, dtype = torch.float32 ) * ( 2.0 * math.pi / self.n_heads )
        grid_init = torch.stack( [ thetas.cos(), thetas.sin() ], -1 )
        grid_init = (
            ( grid_init / grid_init.abs().max( -1, keepdim = True )[0] )
            .view( self.n_heads, 1, 1, 2 )
            .repeat( 1, self.n_levels, self.n_points, 1 )
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter( grid_init.view(-1) )
        constant_( self.attention_weights.weight.data, 0.0 )
        constant_( self.attention_weights.bias.data, 0.0 )
        xavier_uniform_( self.value_proj.weight.data )
        constant_( self.value_proj.bias.data, 0.0 )
        xavier_uniform_( self.output_proj.weight.data )
        constant_( self.output_proj.bias.data, 0.0 )

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        """

        def multi_scale_deformable_attn_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights,
        ):
            """
            References:
                https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
            """
            bs, _, num_heads, embed_dims = value.shape
            _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
            
            value_list = value.split( [ H_ * W_ for H_, W_ in value_spatial_shapes ], dim = 1 ) # split the images patches
            sampling_grids = 2 * sampling_locations - 1 # from 0 -1 to -1 -1
            
            sampling_value_list = []
            for level, (H_, W_) in enumerate(value_spatial_shapes):
                # bs, H_*W_, num_heads, embed_dims ->
                # bs, H_*W_, num_heads*embed_dims ->
                # bs, num_heads*embed_dims, H_*W_ ->
                # bs*num_heads, embed_dims, H_, W_
                value_l_ = value_list[level].flatten( 2 ).transpose( 1, 2 ).reshape( bs * num_heads, embed_dims, H_, W_ )
                # bs, num_queries, num_heads, num_points, 2 ->
                # bs, num_heads, num_queries, num_points, 2 ->
                # bs*num_heads, num_queries, num_points, 2
                sampling_grid_l_ = sampling_grids[:, :, :, level].transpose( 1, 2 ).flatten( 0, 1 )
                # bs*num_heads, embed_dims, num_queries, num_points
                sampling_value_l_ = F.grid_sample(
                    value_l_, sampling_grid_l_, mode = "bilinear", padding_mode = "zeros", align_corners = False
                )
                sampling_value_list.append( sampling_value_l_ )
            # (bs, num_queries, num_heads, num_levels, num_points) ->
            # (bs, num_heads, num_queries, num_levels, num_points) ->
            # (bs, num_heads, 1, num_queries, num_levels*num_points)
            attention_weights = attention_weights.transpose(1, 2).reshape(
                bs * num_heads, 1, num_queries, num_levels * num_points
            )
            output = (
                ( torch.stack( sampling_value_list, dim = -2 ).flatten( -2 ) * attention_weights )
                .sum(-1)
                .view( bs, num_heads * embed_dims, num_queries )
            )
            return output.transpose( 1, 2 ).contiguous()

        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum( s[0] * s[1] for s in value_shapes ) == len_v

        value = self.value_proj( value ) # Project value to n_heads * d_per_head
        if value_mask is not None:
            value = value.masked_fill( value_mask[..., None], float( 0 ) )

        # First estimate x/y offsets for each query
        # Second, estimate attention weights for each query by level and points
        value = value.view( bs, len_v, self.n_heads, self.d_model // self.n_heads ) # N, total_patches, n_heads, d_per_head
        sampling_offsets = self.sampling_offsets( query ).view( bs, len_q, self.n_heads, self.n_levels, self.n_points, 2 ) # N, total_queries, n_heads, n_levels, n_points, 2
        attention_weights = self.attention_weights( query ).view( bs, len_q, self.n_heads, self.n_levels * self.n_points ) # N, total_queries, n_heads, n_levels * n_points
        attention_weights = F.softmax( attention_weights, -1 ).view( bs, len_q, self.n_heads, self.n_levels, self.n_points ) # N, Len_q, n_heads, n_levels, n_points
        # sampling_offsets is used to estimate the sampling locations (x,y) for each query
        
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor( value_shapes, dtype = query.dtype, device = query.device ).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5 # N, total_queries, n_heads, n_levels, n_points, 2
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add # N, total_queries, n_heads, n_levels, n_points, 2
            # Means that for each box with 4 points we have x and y (4,2)
        output = multi_scale_deformable_attn_pytorch( value, value_shapes, sampling_locations, attention_weights )

        return self.output_proj( output )

class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.0, act=nn.ReLU(), n_levels=4, n_points=4):
        
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention( d_model, n_heads, dropout = dropout )
        # self.self_attn = Attention( d_model, n_heads, proj_drop = dropout )
        self.dropout1 = nn.Dropout( dropout )
        self.norm1 = nn.LayerNorm( d_model )

        # Cross attention
        self.cross_attn = MSDeformAttn( d_model, n_levels, n_heads, n_points )
        self.dropout2 = nn.Dropout( dropout )
        self.norm2 = nn.LayerNorm( d_model )

        # FFN
        self.linear1 = nn.Linear( d_model, d_ffn )
        self.act = act
        self.dropout3 = nn.Dropout( dropout )
        self.linear2 = nn.Linear( d_ffn, d_model )
        self.dropout4 = nn.Dropout( dropout )
        self.norm3 = nn.LayerNorm( d_model )

    def forward_ffn(self, tgt):
        
        tgt2 = self.linear2( self.dropout3( self.act( self.linear1( tgt ) ) ) )
        tgt = tgt + self.dropout4( tgt2 )
        return self.norm3( tgt )

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        
        # Self attention
        q = k = embed + query_pos
        tgt = self.self_attn( 
            q.transpose( 0, 1 ), # Sequence first
            k.transpose( 0, 1 ), # Sequence first
            embed.transpose( 0, 1 ), # Sequence first
            attn_mask = attn_mask
        )[0].transpose( 0, 1 ) # B, S, D
        embed = embed + self.dropout1( tgt )
        embed = self.norm1( embed )

        # Cross attention
        tgt = self.cross_attn(
            embed + query_pos, 
            refer_bbox.unsqueeze(2), 
            feats, 
            shapes, 
            padding_mask
        )
        embed = embed + self.dropout2( tgt )
        embed = self.norm2( embed )

        # FFN
        return self.forward_ffn( embed )

class DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        
        super().__init__()
        self.layers = get_clones( decoder_layer, num_layers )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed,  # decoder embeddings
        refer_bbox_logits,  # anchor
        feats,  # image features
        shapes,  # feature shapes
        bbox_head,
        score_head,
        pos_mlp,
        attn_mask=None,
        padding_mask=None,
    ):
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox_logits.sigmoid()
        for i, layer in enumerate( self.layers ):
            
            output = layer( 
                output, 
                refer_bbox, 
                feats, 
                shapes, 
                padding_mask, 
                attn_mask, 
                pos_mlp( refer_bbox ) # MLP for positional encoding based on a estimated bounding box
            )

            bbox = bbox_head[i]( output )
            refined_bbox = torch.sigmoid( bbox + inverse_sigmoid( refer_bbox ) )

            if self.training: # For training phase, we store the prediction for each layer
                dec_cls.append( score_head[i]( output ) )
                if i == 0:
                    dec_bboxes.append( refined_bbox )
                else:
                    dec_bboxes.append( torch.sigmoid( bbox + inverse_sigmoid( last_refined_bbox ) ) )

            elif i == self.eval_idx:
                dec_cls.append( score_head[i]( output ) )
                dec_bboxes.append( refined_bbox )
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack( dec_bboxes ), torch.stack( dec_cls )

class RTDETRDecoder(nn.Module):
    
    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList( nn.Sequential( nn.Conv2d( x, hd, 1, bias = False ), nn.BatchNorm2d( hd ) ) for x in ch )

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer( hd, nh, d_ffn, dropout, act, self.nl, ndp )
        self.decoder = DeformableTransformerDecoder( hd, decoder_layer, ndl, eval_idx )

        # Denoising part
        self.denoising_class_embed = nn.Embedding( nc, hd )

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query: self.tgt_embed = nn.Embedding( nq, hd )
        self.query_pos_head = MLP( 4, 2 * hd, hd, num_layers = 2 )

        # Encoder head
        self.enc_output = nn.Sequential( nn.Linear( hd, hd ), nn.LayerNorm( hd ) )
        self.enc_score_head = nn.Linear( hd, nc )
        self.enc_bbox_head = MLP( hd, hd, 4, num_layers = 3 )

        # Decoder head
        self.dec_score_head = nn.ModuleList( [ nn.Linear( hd, nc ) for _ in range( ndl )])
        self.dec_bbox_head = nn.ModuleList( [ MLP( hd, hd, 4, num_layers = 3 ) for _ in range( ndl ) ] )

        self._reset_parameters()
        self.load_from_pt()

        self.is_warmup = False

    def load_from_pt(self, local='models/pre_trained/rtdetr-c.pth'):

        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        l_state_dict = torch.load( local, map_location = 'cpu', weights_only = False )()
        c_state_dict = self.state_dict()
        l_names = { '.'.join( x.split('.')[3:] ): x  for x in l_state_dict.keys() if int( x.split('.')[2] ) > 31 }
        c_names = [ x for x in c_state_dict.keys() ]
        load_dict = {}
        for l_name in l_names.keys():
            if l_name in c_names:
                c_shape = c_state_dict[l_name].shape
                l_shape = l_state_dict[l_names[l_name]].shape
                if c_shape != l_shape:
                    print( f"Warning: {l_name} shape mismatch: {c_shape} != {l_shape}" )
                    diff = c_shape[0] - l_shape[0]
                    n_value = l_state_dict[l_names[l_name]]
                    if diff > 0:
                        if len( c_shape ) == 2:
                            zero = np.random.normal( n_value.mean().item(), 0.5 * n_value.std().item(), size = [ diff, c_shape[1] ] )
                            zero = torch.from_numpy( zero ).to( n_value.device )
                            n_value = torch.cat( [ zero, n_value ], dim = 0 )
                        elif len( c_shape ) == 1:
                            zero = np.ones( diff ) * bias_cls
                            zero = torch.from_numpy( zero ).to( n_value.device )
                            n_value = torch.cat( [ zero, n_value ], dim = 0 )
                    else:
                        if len( c_shape ) == 2:
                            n_value = n_value[:diff]
                        elif len( c_shape ) == 1:
                            n_value = n_value[:diff]
                    load_dict[l_name] = n_value
                else:
                    load_dict[l_name] = l_state_dict[l_names[l_name]]
            # else:
            #     print( f"Warning: {l_name} not found in current model." )
        self.load_state_dict( load_dict, strict = False )
        print( f"Loaded {len(load_dict)} layers from pre-trained model." )

    def forward(self, x, dn_bbox=None, dn_embed=None, noise_groups=0, num_objects=0):
        
        """
            x = features scales from image -> backbone
            dn_embed = noisy class tokens
            dn_bbox = noisy boxes
            noise_groups = number of groups of noisy queries
            num_objects = number of noisy queries per group
        """

        @torch.no_grad()
        def build_attn_mask(L, M, G, device='cpu'):
            """
            Build an attention mask for decoder queries.
            
            L: Number of learnable queries.
            M: Number of noisy groups.
            G: Number of noisy queries per group.
            Returns:
                attn_mask: A [N, N] tensor with 0 where attention is allowed,
                        and -1e20 where attention should be masked.
            """
            N = L + M * G

            # Create labels: -1 for learnable queries; for noisy queries, assign their group index.
            labels = torch.empty( N, dtype = torch.int64, device = device )
            labels[:L] = -1
            # Compute group indices for noisy queries in a vectorized way.
            labels[L:] = torch.div( torch.arange( N - L, device = device ), G, rounding_mode = 'floor' )

            # Broadcast to compare each pair of queries.
            # Allowed if they have the same label.
            allowed = ( labels.unsqueeze(0) == labels.unsqueeze(1) )

            # Create the final mask: 0 where allowed, -1e20 otherwise.
            attn_mask = torch.where( allowed, torch.tensor( 0.0, device = device ), torch.tensor( -1e20, device = device ) )
            
            return attn_mask

        # Input projection and embedding
        # start = time.time()
        feats, shapes = self._get_encoder_input( x )
        # print( f"Encoder input: {time.time() - start:.2f}s" )

        # start = time.time() 
        queries, refer_bbox, proposal_bboxes, proposal_label_scores = self._get_decoder_input( feats, shapes, dn_embed, dn_bbox )
        # print( f"Decoder input: {time.time() - start:.2f}s" )
        # queries => [ queries, dn_embed ]
        # refer_bbox => [ refer_bbox, dn_bbox ]

        # start = time.time()
        attn_mask = build_attn_mask( self.num_queries, noise_groups, num_objects, device = x[0].device )
        # print( f"Attn mask: {time.time() - start:.2f}s" )
        
        # Decoder
        # start = time.time()
        predicted_bboxes, predicted_label_scores = self.decoder(
            queries,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            # attn_mask = self.attn_mask if self.training else None,
            attn_mask = attn_mask,
        )
        # print( f"Decoder: {time.time() - start:.2f}s" )
        x = predicted_bboxes, predicted_label_scores, proposal_bboxes, proposal_label_scores
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat( ( predicted_bboxes.squeeze(0), predicted_label_scores.squeeze(0).sigmoid() ), -1 )
        return y

    @torch.no_grad()
    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        
        anchors = []
        for i, ( h, w ) in enumerate( shapes ):
            sy = torch.arange( end = h, dtype = dtype, device = device )
            sx = torch.arange( end = w, dtype = dtype, device = device )
            grid_y, grid_x = torch.meshgrid( sy, sx, indexing = "ij" )
            grid_xy = torch.stack( [ grid_x, grid_y ], -1 )  # (h, w, 2)

            valid_WH = torch.tensor( [ w, h ], dtype = dtype, device = device )
            grid_xy = ( grid_xy.unsqueeze(0) + 0.5 ) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like( grid_xy, dtype = dtype, device = device ) * grid_size * ( 2.0**i )
            anchors.append( torch.cat( [ grid_xy, wh ], -1 ).view( -1, h * w, 4 ) )  # (1, h*w, 4)

        anchors = torch.cat( anchors, 1 )  # (1, h*w*nl, 4)
        valid_mask = ( ( anchors > eps ) & ( anchors < 1 - eps ) ).all( -1, keepdim = True )  # 1, h*w*nl, 1
        anchors = torch.log( anchors / ( 1 - anchors ) )
        anchors = anchors.masked_fill( ~valid_mask, float("inf") )
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        
        # Get projection features
        x = [ self.input_proj[i]( feat ) for i, feat in enumerate( x ) ]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append( feat.flatten(2).permute( 0, 2, 1 ) )
            # [nl, 2]
            shapes.append( [ h, w ] )

        # [b, h*w, c]
        feats = torch.cat( feats, 1 )
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_labels=None, dn_bbox=None):
        
        bs = feats.shape[0]

        dn_embed = self.denoising_class_embed( dn_labels ) if dn_labels is not None else None

        # 1) Generate anchors for each location in each feature map
        anchors, valid_mask = self._generate_anchors( shapes, dtype = feats.dtype, device = feats.device )

        # 2) Use an "encoder output" MLP or linear layer
        #    to process the feats => "features" shape [bs, sum(H*W), hidden_dim]
        features = self.enc_output( valid_mask * feats )  # bs, h*w, 256

        # 3) Classification scores for each location => [bs, sum(H*W), nc]
        enc_outputs_scores = self.enc_score_head( features )  # (bs, h*w, nc)

        # 4) From these sum(H*W) locations, pick top-K indices (K = self.num_queries)
        scores, _ = enc_outputs_scores.max( -1 )
        topk_ind = torch.topk( scores, self.num_queries, dim = 1 ).indices.view(-1)
        batch_ind = torch.arange( end = bs, dtype = topk_ind.dtype ).unsqueeze(-1).repeat( 1, self.num_queries ).view(-1)

        # 5) Gather the corresponding top-K features and anchor boxes
        top_k_features = features[batch_ind, topk_ind].view( bs, self.num_queries, -1 )
        top_k_anchors = anchors[:, topk_ind].view( bs, self.num_queries, -1 )

        # 6) Refine anchor => (box + anchor)
        refer_bbox = self.enc_bbox_head( top_k_features ) + top_k_anchors
        proposal_bboxes = refer_bbox.sigmoid()

        # 7) Concatenate the noisy boxes and the anchors
        if dn_bbox is not None:
            refer_bbox = torch.cat( [ refer_bbox, dn_bbox ], 1 )
        
        # 8) Get the scores for the top-K features
        proposal_label_scores = enc_outputs_scores[batch_ind, topk_ind].view( bs, self.num_queries, -1 )

        if self.learnt_init_query:
            queries = top_k_features + self.tgt_embed.weight.unsqueeze(0).repeat( bs, 1, 1 )
        else: 
            queries = top_k_features

        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                queries = queries.detach()
        if dn_embed is not None:
            queries = torch.cat( [ queries, dn_embed ], 1 )

        """
            enc_scores => Try to assign a label for each feature patch
        """        

        return queries, refer_bbox, proposal_bboxes, proposal_label_scores

    def _reset_parameters(self):

        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

        if self.learnt_init_query:
            uniform_( self.tgt_embed.weight, a = 0.001, b = 0.01 )

