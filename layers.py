import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from xformers.ops import memory_efficient_attention

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
    
class DepthwiseConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):

        super( DepthwiseConv, self ).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                in_channels, 
                kernel_size = kernel_size, 
                stride = stride, 
                padding = padding, 
                groups = in_channels
            ), # Depthwise convolution
            nn.ReLU( inplace = True ),
            nn.Conv2d( in_channels, out_channels, kernel_size = 1 ), # Pointwise convolution
            nn.ReLU( inplace = True )
        )

    def forward(self, x):
        x = self.layer( x )
        return x