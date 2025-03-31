import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import ResNet50_Weights

from layers import Attention, MemEffAttention, Mlp, SwiGLUFFNFused, DropPath, LayerScale, DeformableConv2d
from layers import drop_add_residual_stochastic_depth, _build_mlp

from functools import partial
import math

class AttentionBlock(nn.Module):
    
    def __init__( self, dim, num_heads, mlp_ratio=4.0, 
                  qkv_bias=False, proj_bias=True, ffn_bias=True,
                  drop: float=0.0, attn_drop=0.0, init_values=None, drop_path=0.0,
                  act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_class=Attention, ffn_layer=Mlp):
        
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
            in_features = dim,
            hidden_features = mlp_hidden_dim,
            act_layer = act_layer,
            drop = drop,
            bias = ffn_bias,
        )
        self.ls2 = LayerScale( dim, init_values = init_values ) if init_values else nn.Identity()
        self.drop_path2 = DropPath( drop_path ) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x, memory=None, skip_residual=False):
        
        def attn_residual_func(x, memory=None):
            n = self.norm1( x ) # Normalize the input tensor
            if memory is not None: 
                a_out, a, penalty = self.attn.forward_cross( n, memory )
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
                (x, memory) if memory is not None else x,
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
            if not skip_residual: x = x + self.drop_path1( a_out ) # Drop path for the attention block and add to the input tensor
            else: x = self.drop_path1( a_out )

            a_out = ffn_residual_func( x )[0] # Residual function for the MLP block
            x = x + self.drop_path2( a_out ) # Drop path for the MLP block and add to the input tensor
        
        else:

            a_out, a, penalty = attn_residual_func( x, memory ) # Residual function for the attention block
            if not skip_residual: x = x + a_out # Add the residual to the input tensor
            else: x = a_out
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
        skip_first_residual=True,
        attn_class=MemEffAttention,
    ):
        super().__init__()
        norm_layer = partial( nn.LayerNorm, eps = 1e-6 )

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.skip_first_residual = skip_first_residual

        if drop_path_uniform is True:
            dpr = [ drop_path_rate ] * depth
        else:
            dpr = [ x.item() for x in torch.linspace( 0, drop_path_rate, depth ) ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer = Mlp
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
        self.blocks = nn.ModuleList( blocks_list )

        self.norm = norm_layer( embed_dim )

    def forward_features(self, x, memory=None, masks=None):

        if isinstance( x, list ):
            return self.forward_features_list( x, masks )

        attentions = []
        attention_penalty_dict = { k: 0.0 for k in [ "loss_token_cosine_q", "loss_token_cosine_k" ] }
        for i, blk in enumerate( self.blocks ):
            x, a, penalty = blk( x, memory, skip_residual = ( i == 0 ) and self.skip_first_residual )
            attentions.append( a )
            for k in penalty:
                if k in attention_penalty_dict:
                    attention_penalty_dict[k] += penalty[k]            

        x_norm = self.norm( x )
        return {
            "x_norm_cls_token": x_norm[:,0],
            "x_norm_patch_tokens": x_norm[:,1:],
            "attn": attentions,
            "attention_penalty": attention_penalty_dict,
        }

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret

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

class Backbone(nn.Module):

    def __init__(self, out_channels=384, model="resnet50"):
        super(Backbone, self).__init__()
        if model == "resnet50":
            self.body = FrozenResNetBackbone( out_channels = out_channels )
        elif model == "simple":
            self.body = SimpleBackbone( out_channels = out_channels )
        else:
            raise NotImplementedError(f"Backbone model {model} not implemented.")
    
    def forward(self, x):
        return self.body( x )

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

class HGSystem(nn.Module):
    """
      1) x => Conv + ReLU => h1
      2) h1 => MaxPool => h2
      3) h1 => Conv + ReLU => h3
      4) cat(h2, h3) => h4
      5) h4 => Conv + ReLU => h5
    """
    def __init__(self, in_channels, mid_channels, out_channels):

        super(HGSystem, self).__init__()

        # Step 1: input => conv + relu => h1
        self.step1 = nn.Sequential(
            nn.Conv2d( in_channels, mid_channels, kernel_size = 3, padding = 1 ),
            nn.ReLU( inplace = True )
        )
        self.pool = nn.MaxPool2d( kernel_size = 2, stride = 2 )

        # Step 3: from h1 => conv + ReLU => h3
        self.step2 = nn.Sequential(
            nn.Conv2d( mid_channels, mid_channels, kernel_size = 3, padding = 1 ),
            nn.ReLU( inplace = True )
        )

        # Step 5: from cat(h2, h3) => conv + ReLU => h5
        self.step3 = nn.Sequential(
            nn.Conv2d( 2*mid_channels, out_channels, kernel_size = 3, padding = 1 ),
            nn.ReLU( inplace = True )
        )

    def forward(self, x):

        h1 = self.step1( x )
        h2 = self.pool( h1 )  # MaxPool
        h3 = self.step2( h1 )  # Conv + ReLU
        h4 = torch.cat( [ h2, h3 ], dim = 1 )  # Concatenate along channel dimension
        h5 = self.step3( h4 )  # Conv + ReLU

        return h5

class HGBlock(nn.Module):
    """
      1) h => Conv + ReLU => h1
      2) h1 => Conv + ReLU => h2
      3) h2 => Conv + ReLU => h3
      4) h3 => Conv + ReLU => h4
      5) h4 => Conv + ReLU => h5
      6) h5 => Conv + ReLU => h6
      7) cat(h, h1, h2, h3, h4, h5) => h7
      8) h7 => Conv + ReLU => h8
      9) h8 => Conv + ReLU => h9
    """
    def __init__(self, channels, num_layers=6):

        super(HGBlock, self).__init__()

        # layers
        self.layers = nn.ModuleList()
        for i in range( num_layers ):
            self.layers.append( nn.Sequential(
                nn.Conv2d( channels, channels, 3, padding = 1 ),
                nn.ReLU( inplace = True )
            ) )

        self.final_conv = nn.Sequential(
            nn.Conv2d( num_layers*channels, channels, 3, padding = 1 ),
            nn.ReLU( inplace = True ),
            nn.Conv2d( channels, channels, 3, padding = 1 ),
            nn.ReLU( inplace = True )
        )

    def forward(self, h):
        hs = []
        for layer in self.layers:
            h = layer( h )
            hs.append( h )
        h = torch.cat( hs, dim = 1 )
        h = self.final_conv( h )
        return h

class RepC3(nn.Module):
    """
      1) h => Conv + ReLU => h1
      2) h1 => Conv + SiLU => h2
      3) h2 => Conv + SiLU => h3
      4) h3 => Conv + SiLU => h4
      5) h => Conv + ReLU => h5
      6) cat(h4, h5) => h6
    """
    def __init__(self, in_channels, out_channels):

        super(RepC3, self).__init__()

        self.stage0 = nn.Sequential(

            nn.Conv2d( in_channels, out_channels, 3, padding = 1 ),
            nn.ReLU( inplace = True ),

            nn.Conv2d( out_channels, out_channels, 3, padding = 1 ),
            nn.SiLU(),

            nn.Conv2d( out_channels, out_channels, 3, padding = 1 ),
            nn.SiLU(),

            nn.Conv2d( out_channels, out_channels, 3, padding = 1 ),
            nn.SiLU()
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d( in_channels, out_channels, 1, padding = 0 ),
            nn.ReLU( inplace = True )
        )

    def forward(self, x):
        
        # main branch        
        h = self.stage0( x )

        # shortcut
        hx = self.shortcut( x )

        # cat => final
        h6 = torch.cat( [ h, hx ] , dim = 1 )

        return h6

class LinearEncoder(nn.Module):
    """
    Overall architecture:

    image => HGSyste[0] => h1
    h1 => HGBlock[1] => h2
    h2 => DWConv[2] => h3
    h3 => HGBlock[3] => h4
    h4 => HGBlock[4] => h5
    h5 => DWConv[5] => h6
    h6 => HGBlock[6] => h7
    h7 => HGBlock[7] => h8
    h8 => HGBlock[8] => h9
    h9 => HGBlock[9] => h10
    h10 => HGBlock[10] => h11
    h11 => DWConv[11] => h12
    h12 => HGBlock[12] => h13
    h13 => HGBlock[13] => h14
    h14 => Conv => h15
    h15 => AIFI[15] => h16
    h16 => Conv => h17
    h17 => Upsample => h18

    h11 => Conv => h19
    cat(h18, h19) => h20
    h20 => RepC3[20] => h21
    h21 => Conv => h22
    h22 => Upsample => h23
    h23 => Conv => h24

    h5 => Conv => h25
    cat(h24, h25) => h26
    h26 => RepC3[25] => h27
    h27 => Conv => h28

    cat(h22, h28) => h29
    h29 => RepC3[28] => h30
    h30 => Conv => h31

    cat(h17, h31) => h32
    h32 => RepC3[31] => h33

    outputs = [h27, h30, h33]

    Shapes expected: 80x80xN, 40x40xN, 20x20xN
    """
    def __init__(self, in_channels=3, base_ch=32):
        super(Encoder, self).__init__()

        # HGSyste[0]
        self.hgs0 = HGSystem(in_channels, base_ch, base_ch)

        # HGBlock[1], [3], [4], ...
        self.hgb1 = HGBlock(base_ch)
        self.dw2   = DepthwiseConv(base_ch, base_ch)
        self.hgb3 = HGBlock(base_ch)
        self.hgb4 = HGBlock(base_ch)
        self.dw5   = DepthwiseConv(base_ch, base_ch)
        self.hgb6 = HGBlock(base_ch)
        self.hgb7 = HGBlock(base_ch)
        self.hgb8 = HGBlock(base_ch)
        self.hgb9 = HGBlock(base_ch)
        self.hgb10 = HGBlock(base_ch)

        self.dw11  = DepthwiseConv(base_ch, base_ch)
        self.hgb12 = HGBlock(base_ch)
        self.hgb13 = HGBlock(base_ch)

        # h14 => Conv => h15
        self.conv14_15 = nn.Conv2d(base_ch, base_ch, 3, padding=1)

        # h15 => AIFI[15] => h16
        # Suppose we want the same “base_ch” in/out
        self.aifi15 = AIFI(base_ch, embed_dim=base_ch)  

        # h16 => Conv => h17
        self.conv16_17 = nn.Conv2d(2*base_ch, base_ch, 3, padding=1) 
        # ^ Notice that AIFI’s output is cat(...) of dimension (2 * base_ch).
        #   So we go from 2*base_ch => base_ch. Adjust as needed.

        # h17 => Upsample => h18
        self.up17_18 = nn.Upsample(scale_factor=2, mode='nearest')

        # h11 => Conv => h19
        self.conv11_19 = nn.Conv2d(base_ch, base_ch, 1)

        # cat(h18, h19) => h20
        # h20 => RepC3[20] => h21
        self.rep20 = RepC3(in_channels=2*base_ch, out_channels=base_ch)

        # h21 => Conv => h22
        self.conv21_22 = nn.Conv2d(2*base_ch, base_ch, 3, padding=1)

        # h22 => Upsample => h23
        self.up22_23 = nn.Upsample(scale_factor=2, mode='nearest')

        # h23 => Conv => h24
        self.conv23_24 = nn.Conv2d(base_ch, base_ch, 3, padding=1)

        # h5 => Conv => h25
        self.conv5_25 = nn.Conv2d(base_ch, base_ch, 1)

        # cat(h24, h25) => h26
        # h26 => RepC3[25] => h27
        self.rep25 = RepC3(in_channels=2*base_ch, out_channels=base_ch)

        # h27 => Conv => h28
        self.conv27_28 = nn.Conv2d(2*base_ch, base_ch, 3, padding=1)

        # cat(h22, h28) => h29
        # h29 => RepC3[28] => h30
        self.rep28 = RepC3(in_channels=2*base_ch, out_channels=base_ch)

        # h30 => Conv => h31
        self.conv30_31 = nn.Conv2d(2*base_ch, base_ch, 3, padding=1)

        # cat(h17, h31) => h32
        # h32 => RepC3[31] => h33
        self.rep31 = RepC3(in_channels=2*base_ch, out_channels=base_ch)

    def forward(self, x):
        # 1) image => HGSyste[0] => h1
        h1 = self.hgs0(x)
        # 2) h1 => HGBlock[1] => h2
        h2 = self.hgb1(h1)
        # 3) h2 => DWConv[2] => h3
        h3 = self.dw2(h2)
        # 4) h3 => HGBlock[3] => h4
        h4 = self.hgb3(h3)
        # 5) h4 => HGBlock[4] => h5
        h5 = self.hgb4(h4)
        # 6) h5 => DWConv[5] => h6
        h6 = self.dw5(h5)
        # 7) h6 => HGBlock[6] => h7
        h7 = self.hgb6(h6)
        # 8) h7 => HGBlock[7] => h8
        h8 = self.hgb7(h7)
        # 9) h8 => HGBlock[8] => h9
        h9 = self.hgb8(h8)
        # 10) h9 => HGBlock[9] => h10
        h10 = self.hgb9(h9)
        # 11) h10 => HGBlock[10] => h11
        h11 = self.hgb10(h10)
        # 12) h11 => DWConv[11] => h12
        h12 = self.dw11(h11)
        # 13) h12 => HGBlock[12] => h13
        h13 = self.hgb12(h12)
        # 14) h13 => HGBlock[13] => h14
        h14 = self.hgb13(h13)
        # 15) h14 => Conv => h15
        h15 = self.conv14_15(h14)
        # 16) h15 => AIFI[15] => h16
        h16 = self.aifi15(h15)
        # 17) h16 => Conv => h17
        h17 = self.conv16_17(h16)
        # 18) h17 => Upsample => h18
        h18 = self.up17_18(h17)

        # 19) h11 => Conv => h19
        h19 = self.conv11_19(h11)
        # 20) cat(h18, h19) => h20
        h20 = torch.cat([h18, h19], dim=1)
        # 21) h20 => RepC3[20] => h21
        h21 = self.rep20(h20)
        # 22) h21 => Conv => h22
        h22 = self.conv21_22(h21)
        # 23) h22 => Upsample => h23
        h23 = self.up22_23(h22)
        # 24) h23 => Conv => h24
        h24 = self.conv23_24(h23)
        # 25) h5 => Conv => h25
        h25 = self.conv5_25(h5)
        # 26) cat(h24, h25) => h26
        h26 = torch.cat([h24, h25], dim=1)
        # 27) h26 => RepC3[25] => h27
        h27 = self.rep25(h26)
        # 28) h27 => Conv => h28
        h28 = self.conv27_28(h27)
        # 29) cat(h22, h28) => h29
        h29 = torch.cat([h22, h28], dim=1)
        # 30) h29 => RepC3[28] => h30
        h30 = self.rep28(h29)
        # 31) h30 => Conv => h31
        h31 = self.conv30_31(h30)
        # 32) cat(h17, h31) => h32
        h32 = torch.cat([h17, h31], dim=1)
        # 33) h32 => RepC3[31] => h33
        h33 = self.rep31(h32)

        # final output = [h27, h30, h33]
        return [h27, h30, h33]

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

class PredictorHead(nn.Module):

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
            nn.Linear( feature_dim+64, 384 ),
            nn.ReLU(),
            nn.Dropout( 0.1 ),
            nn.Linear( 384, 384 ),
            nn.ReLU(),
            nn.Dropout( 0.1 ),
            nn.Linear( 384, 4 )
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear( feature_dim+64, 384 ),
            nn.ReLU(),
            nn.Dropout( 0.1 ),
            nn.Linear( 384, 384 ),
            nn.ReLU(),
            nn.Dropout( 0.1 ),
            nn.Linear( 384, num_classes )
        )

    def forward(self, features):

        class_logits = self.cls_head( features )  # Shape: [batch_size, seq_len, num_classes]
        boxe_values = self.bbox_head( features ).sigmoid()  # Shape: [batch_size, seq_len, 4]
        return boxe_values, class_logits

class TransformerDetection(nn.Module):
    
    def __init__(self, feature_dim, num_heads, num_layers, num_detections, num_bins, num_classes):
    
        super( TransformerDetection, self ).__init__()

        self.num_detections = num_detections
    
        self.tokens_small_objects = nn.Parameter( torch.randn( 1, num_detections // 3, feature_dim + 64 ), requires_grad = True )
        self.tokens_medium_objects = nn.Parameter( torch.randn( 1, num_detections // 3, feature_dim + 64 ), requires_grad = True )
        self.tokens_large_objects = nn.Parameter( torch.randn( 1, num_detections // 3, feature_dim + 64 ), requires_grad = True )

        self.decoder = Transformer( embed_dim = feature_dim + 64, depth = num_layers, num_heads = num_heads, skip_first_residual = False, attn_class = Attention )
        self.pos_embed = nn.Parameter( torch.randn( 1, 4096, 64 ) )  # Positional embeddings

        self.small_detection_head = DetectionHead( feature_dim, num_classes, num_bins )  # Detection head
        self.medium_detection_head = DetectionHead( feature_dim, num_classes, num_bins )
        self.large_detection_head = DetectionHead( feature_dim, num_classes, num_bins )

        self.interpolate_offset = 0.1
        self.interpolate_antialias = True

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

    def prepare_tokens_with_masks(self, x, w, h):
        bs = x.shape[0]
        pos = self.interpolate_pos_encoding( w, h, x ) # [ B HW D ] Positional encoding
        x = torch.cat( [ x, pos.expand( bs, -1, -1 ) ], dim = -1 ) # [ B HW+1 D ] Add class token
        return x

    def forward(self, encoder_outputs):

        bs = encoder_outputs[0].shape[0]
        
        # Prepare the KV cahce from the encoder
        small_objects_memory = self.prepare_tokens_with_masks( encoder_outputs[0], 64, 64 )
        medium_objects_memory = self.prepare_tokens_with_masks( encoder_outputs[1], 32, 32 )
        large_objects_memory = self.prepare_tokens_with_masks( encoder_outputs[2], 16, 16 )

        # Decode every object
        small_objects_decoder_outputs = self.decoder(
            x = self.tokens_small_objects.expand( bs, -1, -1 ),
            memory = small_objects_memory, # Only the largest scale feature map
        )
        medium_objects_decoder_outputs = self.decoder(
            x = self.tokens_medium_objects.expand( bs, -1, -1 ),
            memory = torch.cat( [ small_objects_decoder_outputs['x_norm_patch_tokens'], medium_objects_memory ], dim = 1 ), # Medium scale feature map + detected small objects
            # memory = medium_objects_memory # Medium scale feature map + detected small objects
        )
        large_objects_decoder_outputs = self.decoder(
            x = self.tokens_large_objects.expand( bs, -1, -1 ),
            memory = torch.cat( [ small_objects_decoder_outputs['x_norm_patch_tokens'], medium_objects_decoder_outputs['x_norm_patch_tokens'], large_objects_memory ], dim = 1 ), # Small scale feature map + detected small and medium objects
            # memory = large_objects_memory, # Small scale feature map + detected small and medium objects
        )
        
        # Detection Head
        small_bboxes, small_class_logits = self.small_detection_head( small_objects_decoder_outputs['x_norm_patch_tokens'] ) # Detect small objects
        medium_bboxes, medium_class_logits = self.medium_detection_head( medium_objects_decoder_outputs['x_norm_patch_tokens'] ) # Detect medium objects
        large_bboxes, large_class_logits = self.large_detection_head( large_objects_decoder_outputs['x_norm_patch_tokens'] ) # Detect large objects

        class_logits = torch.cat( [ small_class_logits, medium_class_logits, large_class_logits ], dim = 1 )  # [bs, num_tokens, num_classes]
        bboxes = torch.cat( [ small_bboxes, medium_bboxes, large_bboxes ], dim = 1 )  # [bs, num_tokens, 4]
      
        return bboxes, class_logits
