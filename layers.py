import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import ResNet50_Weights

from xformers.ops import memory_efficient_attention

from functools import partial
import math

@torch.no_grad()
def build_self_and_memory_only_mask(n, m, device):
    
    # Start with full -inf (blocked)
    mask = torch.full( ( 1, 1, n, m + n ), float( '-inf' ), device = device )

    # Allow attention to all memory
    mask[:, :, :, :m] = 0

    # Allow each x_i to attend to itself (at position m + i)
    for i in range(n):
        mask[:, :, i, m + i] = 0

    return mask

class Attention(nn.Module):
    
    def __init__(self, dim, 
                 num_heads=8, 
                 qkv_bias=False, 
                 proj_bias=True,
                 attn_drop=0.0, 
                 proj_drop=0.0,
                 penalty_types=None,
                 min_std=0.05,
                 alpha_var_floor=1.0,
                 weight_qk=0.1):
        
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear( dim, dim * 3, bias = qkv_bias )
        self.attn_drop = nn.Dropout( attn_drop )
        self.proj = nn.Linear( dim, dim, bias = proj_bias )
        self.proj_drop = nn.Dropout( proj_drop )
        
        # Store penalty configs
        self.penalty_types = penalty_types or ["loss_variance_q", "loss_variance_k", "loss_covar_q", "loss_covar_k", "loss_orth_qw"]  # e.g. ["variance_q", "covar_k", "orth_qw"]
        self.min_std = min_std
        self.alpha_var_floor = alpha_var_floor
        self.weight_qk = weight_qk

    def compute_token_cosine_penalty(self, features):
        """
        Computes a penalty for high cosine similarity between token embeddings across tokens.

        Args:
            features: [bs, N, C/H] - Token embeddings for a single head across batch and tokens.

        Returns:
            penalty: Scalar penalty for cosine similarity between token embeddings.
        """
        # Normalize features along the embedding dimension
        normed_features = F.normalize( features, dim = -1 )  # [bs, N, C/H]

        # Compute cosine similarity for all tokens in a batch
        cosine_sim = torch.einsum( "bnf,bmf->bnm", normed_features, normed_features )  # [bs, N, N]

        # Create a mask for the diagonal elements
        N = cosine_sim.size( 1 )  # Number of tokens
        bs = cosine_sim.size( 0 )  # Batch size
        off_diag_mask = ~torch.eye( N, device = cosine_sim.device ).bool()  # [N, N]
        off_diag_mask = off_diag_mask.unsqueeze(0).expand( bs, -1, -1 )

        # Extract and average the off-diagonal elements
        off_diag_sim = cosine_sim[off_diag_mask].view( cosine_sim.size(0), -1 )  # [bs, N*(N-1)]
        mean_off_diag_sim = off_diag_sim.mean()  # Scalar penalty

        return mean_off_diag_sim

    def forward(self, x, attn_bias=None):
        
        B, N, C = x.shape

        qkv = self.qkv( x ) # [ B N C ] -> [ B N 3(qkv) * C ]
        qkv = qkv.reshape( B, N, 3, self.num_heads, C // self.num_heads ) # [ B N 3(qkv) * C ] -> [ B N 3(qkv) H C/H ]
        qkv = qkv.permute( 2, 0, 3, 1, 4 ) # [ B N 3(qkv) H C/H ] -> [ 3(qkv) B H N C/H ]

        # q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose( -1, -2 )

        if attn_bias:
            attn = attn + attn_bias

        attn = attn.softmax( dim = -1 )
        attn_d = self.attn_drop( attn )

        x = ( attn_d @ v ).transpose( 1, 2 ).reshape( B, N, C )
        x = self.proj( x )
        x = self.proj_drop( x )

        return ( x, attn, {} )
    
    def forward_cross(self, x, memory, attn_bias=None):
        
        B, N, C = x.shape

        # 1) Compute Q, K, V all at once
        head_dim = C // self.num_heads
        qkv = self.qkv( x ) # [ B N C ] -> [ B N 3(qkv) * C ]
        qkv = qkv.reshape( B, N, 3, self.num_heads, head_dim ) # [ B N 3(qkv) * C ] -> [ B N 3(qkv) H C/H ]
        qkv = qkv.permute( 2, 0, 3, 1, 4 ) # [ B N 3(qkv) H C/H ] -> [ 3(qkv) B H N C/H ]
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        # 2) Memory compute Q, K, V all at once
        _, N_mem, _ = memory.shape
        qkv_mem = self.qkv( memory ) # [ B N_mem C ] -> [ B N_mem 3(qkv) * C ]
        qkv_mem = qkv_mem.reshape( B, N_mem, 3, self.num_heads, head_dim ) # [ B N_mem 3(qkv) * C ] -> [ B N_mem 3(qkv) H C/H ]
        qkv_mem = qkv_mem.permute( 2, 0, 3, 1, 4 ) # [ B N_mem 3(qkv) H C/H ] -> [ 3(qkv) B H N_mem C/H ]
        k_mem, v_mem = qkv_mem[1], qkv_mem[2]

        # 3) Concatenate memory tokens with input tokens
        k = torch.cat( ( k_mem, k ), dim = -2 ) # [ B H N C/H ] -> [ B H (N + N_mem) C/H ]
        v = torch.cat( ( v_mem, v ), dim = -2 ) # [ B H N C/H ] -> [ B H (N + N_mem) C/H ]

        # 4) Compute attention map
        attn = q @ k.transpose( -1, -2 )
        if not attn_bias is None:
            attn = attn + attn_bias

        attn = attn.softmax( dim = -1 )
        attn_d = self.attn_drop( attn )

        # 5) Compute output
        x = ( attn_d @ v ).transpose( 1, 2 ).reshape( B, N, C )
        x = self.proj( x )
        x = self.proj_drop( x )

        return ( x, attn, {} )

class MemEffAttention(Attention):

    def forward(self, x: torch.Tensor, attn_bias=None):
        """
        Args:
          x: [B, N, C]
          attn_bias: optional bias for attention computation
        Returns:
          output: [B, N, C]  (the attended features)
          penalties_dict: dict of penalty_name -> penalty_value
        """
        B, N, C = x.shape

        # 1) Compute Q, K, V all at once
        head_dim = C // self.num_heads
        qkv = self.qkv( x )  # [B, N, 3*C]
        qkv = qkv.reshape( B, N, 3, self.num_heads, head_dim )  # [B, N, 3, H, C/H]
        qkv = qkv.permute( 2, 0, 1, 3, 4 )  # [3, B, N, H, C/H]

        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, N, H, C/H]

        # 2) Memory-efficient attention
        #    (assumes you have a function memory_efficient_attention(q, k, v, ...))
        x = memory_efficient_attention( q, k, v, attn_bias = attn_bias )  # [B, N, H, C/H]
        x = x.reshape( B, N, C )

        # 3) Final projection
        x = self.proj_drop( self.proj( x ) )
        
        return ( x, None, {} )

    def forward_cross(self, x, memory, attn_bias=None):

        B, N, C = x.shape

        # 1) Compute Q, K, V all at once
        head_dim = C // self.num_heads
        qkv = self.qkv( x )
        qkv = qkv.reshape( B, N, 3, self.num_heads, head_dim )
        qkv = qkv.permute( 2, 0, 1, 3, 4 )

        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2) Memory compute Q, K, V all at once
        _, N_mem, _ = memory.shape
        qkv_mem = self.qkv( memory )
        qkv_mem = qkv_mem.reshape( B, N_mem, 3, self.num_heads, head_dim )
        qkv_mem = qkv_mem.permute( 2, 0, 1, 3, 4 )

        k_mem, v_mem = qkv_mem[1], qkv_mem[2]

        # 3) Concatenate memory tokens with input tokens
        k = torch.cat( ( k_mem, k ), dim = 1 )
        v = torch.cat( ( v_mem, v ), dim = 1 )

        # 4) Compute attention map
        x = memory_efficient_attention( q, k, v, attn_bias = attn_bias )
        x = x.reshape( B, N, C )

        # 5) Final projection
        x = self.proj_drop( self.proj( x ) )

        return ( x, None, {} )

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, bias=True):
        
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear( in_features, hidden_features, bias = bias )
        self.act = act_layer()
        self.fc2 = nn.Linear( hidden_features, out_features, bias = bias )
        self.drop = nn.Dropout( drop )

    def forward(self, x):

        x = self.fc1( x )
        x = self.act( x )
        x = self.drop( x )
        x = self.fc2( x )
        x = self.drop( x )
        
        return x

class LayerScale(nn.Module):
    
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        
        self.inplace = inplace
        self.gamma = nn.Parameter( init_values * torch.ones( dim ) )

    def forward(self, x):
        return x.mul_( self.gamma ) if self.inplace else x * self.gamma
    
def drop_path(x, drop_prob=0.0, training=False):
    
    # x => [ B, N, D ]
    if drop_prob == 0.0 or not training:
        return x
    
    with torch.no_grad():
        keep_prob = 1 - drop_prob
        shape = ( x.shape[0], ) + ( 1, ) * ( x.ndim - 1 )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty( shape ).bernoulli_( keep_prob ) # [ B, 1, 1, 1 ] with values in {0, 1}
        if keep_prob > 0.0:
            random_tensor.div_( keep_prob )

    output = x * random_tensor

    return output

class DropPath(nn.Module):

    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path( x, self.drop_prob, self.training )

class SwiGLUFFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=None, drop=0.0, bias=True):
        
        super().__init__()
        
        # Default hidden and output dimensions if not provided
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Linear layer to map input to twice the hidden features (split for gating)
        self.w12 = nn.Linear( in_features, 2 * hidden_features, bias = bias )

        # Linear layer to map gated hidden features to output features
        self.w3 = nn.Linear( hidden_features, out_features, bias = bias )

    def forward(self, x):
        
        # First linear transformation: produce a tensor with 2x hidden features
        x12 = self.w12( x )

        # Split the tensor into two equal parts along the last dimension
        x1, x2 = x12.chunk( 2, dim = -1 )

        # Apply the SwiGLU activation: SiLU(x1) * x2 (element-wise gating)
        hidden = F.silu( x1 ) * x2

        # Project the gated hidden features to the output dimension
        return self.w3( hidden )

class SwiGLUFFNFused(SwiGLUFFN):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=None, drop=0.0, bias=True):
                
        # Default hidden and output dimensions if not provided
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Adjust hidden_features for efficient memory alignment:
        # - Multiply by 2/3 for dimensionality reduction.
        # - Round up to the nearest multiple of 8 for hardware efficiency (e.g., Tensor Cores).
        # Example: hidden_features = 100
        # hidden_features = ( 100 * 2 / 3 ) = 66.67 -> This scaling factor is likely chosen to reduce the number of parameters and computational overhead in the feed-forward network.
        # hidden_features = 66.67 + 7 = 73.67 -> This is likely done to ensure that the number of hidden features is a multiple of 8 for hardware efficiency.
        # hidden_features = 73.67 // 8 * = 9 -> This is done to round the number of hidden features to the nearest multiple of 8.
        # hidden_features = 9 * 8 = 72 -> This is the final number of hidden features used in the feed-forward network.
        hidden_features = ( int( hidden_features * 2 / 3 ) + 7) // 8 * 8

        # Call the parent constructor with adjusted hidden features
        super().__init__( in_features = in_features, hidden_features = hidden_features, out_features = out_features, bias = bias )

def drop_add_residual_stochastic_depth(x, residual_func, sample_drop_ratio=0.0):
    
    """
    Applies stochastic depth to a batch of data by randomly dropping a subset of the batch,
    applying a residual function to the retained subset, and adding the scaled residual
    back to the original input tensor.

    Parameters:
        x (torch.Tensor): Input tensor of shape (b, n, d), where:
            - b: Batch size
            - n: Sequence length or number of features
            - d: Feature dimension
        residual_func (function): A function that computes a residual from the input subset.
        sample_drop_ratio (float): Fraction of the batch to drop (0.0 means no drop, 1.0 means full drop).

    Returns:
        torch.Tensor: Updated tensor of the same shape as the input (b, n, d).
    """
    
    # Step 1: Get dimensions of the input tensor
    if type( x ) == tuple:
        x, memory = x
    else: memory = None

    b, n, d = x.shape  # b: batch size, n: sequence length, d: feature dimension
    
    # Calculate how many samples to retain (at least 1 sample is always retained)
    sample_subset_size = max( int( b * ( 1 - sample_drop_ratio ) ), 1 )
    
    # Randomly permute the indices of the batch and select a subset
    brange = torch.randperm( b, device = x.device )[:sample_subset_size]
    x_subset = x[brange]  # Select only the samples corresponding to the subset
    
    # Step 2: Apply the residual function to the subset
    # This produces a residual tensor for the retained samples
    if memory is not None:
        memory_subset = memory[brange]
        residual, a, penalty = residual_func( x_subset, memory_subset )  # Shape: ( subset_size, n, d )
    else:
        residual, a, penalty = residual_func( x_subset )   # Shape: ( subset_size, n, d )
    
    # Flatten input and residual tensors along all dimensions except the batch dimension
    # This simplifies tensor operations when adding the residual back
    x_flat = x.flatten(1)  # Shape: (b, n * d)
    residual = residual.flatten(1)  # Shape: (subset_size, n * d)

    # Compute a scaling factor for the residual to account for the reduced subset size
    residual_scale_factor = b / sample_subset_size

    # Step 3: Add the scaled residual back to the original tensor
    # torch.index_add:
    #   - Adds the residual tensor at specific indices (brange) in the original tensor (x_flat)
    #   - Scales the residual with `alpha` to maintain proper magnitude
    x_plus_residual = torch.index_add(
        x_flat,                         # Original flattened tensor
        0,                              # Dimension along which to add (batch dimension)
        brange,                         # Indices of the retained samples
        residual.to( dtype = x.dtype ), # Residual tensor converted to match input dtype
        alpha = residual_scale_factor   # Scaling factor for the residual
    )

    # Step 4: Reshape the updated tensor back to the original shape
    return x_plus_residual.view_as(x), a, penalty  # Shape: (b, n, d)

class Block(nn.Module):
    
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
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=0,
        skip_first_residual=True,
        attn_class=MemEffAttention,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
        """

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
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append( [ nn.Identity() ] * i + blocks_list[ i : i + chunksize ] )
            self.blocks = nn.ModuleList( [ BlockChunk( p ) for p in chunked_blocks ] )
        else:
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

class DeformableConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
    
        super().__init__()
    
        self.offset_conv = nn.Conv2d( in_ch, 
                                      2*kernel_size*kernel_size, 
                                      kernel_size, 
                                      padding = padding, 
                                      stride = stride )
        
        self.mask_conv = nn.Conv2d( in_ch, 
                                    kernel_size*kernel_size, 
                                    kernel_size, 
                                    padding = padding, 
                                    stride = stride )
        
        self.weight = nn.Parameter( torch.randn( out_ch, in_ch, kernel_size, kernel_size ) )
        self.bias = nn.Parameter( torch.zeros( out_ch ) )

    def forward(self, x):
        offset = self.offset_conv( x ) # 2 Kernels for 2D offset
        mask = torch.sigmoid( self.mask_conv( x ) ) # 1 Kernel for mask
        return deform_conv2d( input = x, offset = offset, weight = self.weight,
                              bias = self.bias, padding = 1, mask = mask )

class SimpleBackbone(nn.Module):

    def __init__(self, in_channels=3):
    
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
            DeformableConv2d( 256, 384, kernel_size = 3, padding = 1, stride = 1 ),
            nn.BatchNorm2d( 384 ),
            nn.ReLU( inplace = True )
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d( 256, 512, kernel_size = 3, stride = 2, padding = 1 ),  # 64->32 - 1/16 scale
            nn.BatchNorm2d( 512 ),
            nn.ReLU( inplace = True )
        ) # 32x32
        self.def_stage4 = nn.Sequential( 
            DeformableConv2d( 512, 384, kernel_size = 3, padding = 1, stride = 1 ),
            nn.BatchNorm2d( 384 ),
            nn.ReLU( inplace = True )
        )

        self.stage5 = nn.Sequential(
            nn.Conv2d( 512, 512, kernel_size = 3, stride = 2, padding = 1 ),  # 32->16 - 1/32 scale
            nn.BatchNorm2d( 512 ),
            nn.ReLU( inplace = True )
        ) # 16x16
        self.def_stage5 = nn.Sequential( 
            DeformableConv2d( 512, 384, kernel_size = 3, padding = 1, stride = 1 ),
            nn.BatchNorm2d( 384 ),
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
    
    def __init__(self, pretrained=True, in_channels = 3, out_channels=384):
    
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

    def __init__(self, in_channels=3, out_channels=384, use_resnet=False):
        super(Backbone, self).__init__()
        if use_resnet:
            self.body = FrozenResNetBackbone( in_channels=in_channels, out_channels=out_channels )
        else:
            self.body = SimpleBackbone( in_channels=in_channels )
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

class VisionEncoder(nn.Module):

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

# ----------------------------------------------------------------------------
# Prediction Part   
# ----------------------------------------------------------------------------

class ProposalHead(nn.Module):

    def __init__(self, input_dim):

        super( ProposalHead, self ).__init__()

        self.net = nn.Sequential(
            nn.Linear( input_dim, input_dim ),
            nn.ReLU(),
            nn.Dropout( 0.1 ),
            nn.Linear( input_dim, input_dim ),
            nn.ReLU(),
            nn.Dropout( 0.1 ),
            nn.Linear( input_dim, 1 )
        )

    def forward(self, x):
        x = self.net( x )
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

        # # Bounding Box Top-Left Prediction
        # bboxes_top_logits = self.bbox_head( features )  # shape: [B, T, 200]
        # ( cx_logits, cy_logits,
        #   width_logits, height_logits ) = bboxes_top_logits.split( self.num_bins, dim = -1 )  # each: [B, T, 100]
        
        # # Convert logits to probabilities
        # cx_probs = F.softmax( cx_logits, dim = -1 )  # [B, T, 100]
        # cy_probs  = F.softmax( cy_logits, dim = -1 )   # [B, T, 100]
        # width_probs  = F.softmax( width_logits, dim = -1 )  # [B, T, 100]
        # height_probs = F.softmax( height_logits, dim = -1 )
        
        # # Define discrete bins uniformly over [0, 1]
        # bins = torch.linspace( 0, 1, steps = self.num_bins, device = cy_logits.device ).view( 1, 1, self.num_bins )  # [1, 1, 100]
        
        # # Compute the expected value for top and left coordinates
        # cx_value = torch.sum( cx_probs * bins, dim = -1 )   # shape: [B, T]
        # cy_value  = torch.sum( cy_probs  * bins, dim = -1 )  # shape: [B, T]
        # width_value  = torch.sum( width_probs  * bins, dim = -1 ) # [B, T]
        # height_value = torch.sum( height_probs * bins, dim = -1 ) # [B, T]

        # # Continuous bounding box (normalized): [top, left, width, height]
        # bbox_cont = torch.stack( [ cx_value, cy_value, width_value, height_value ], dim = -1 )

        # # Discrete bounding box (normalized): [top, left, width, height]
        # bbox_logits =  ( cx_logits, cy_logits, width_logits, height_logits )

        # return ( bbox_cont, bbox_logits ), class_logits

# Transformer Decoder with Cross-Attention
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