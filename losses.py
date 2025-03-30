import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import math

def cross_entropy(logits, targets, temp=1.0, bw_inplace=False, reduction='mean'):
    """
    Scaled cross-entropy loss with temperature scaling and optional in-place backward computation.

    Args:
        logits (torch.Tensor): Unnormalized scores (logits) of shape (N, C).
        targets (torch.Tensor): Ground truth labels of shape (N,) with values in [0, C-1].
        temp (float): Temperature scaling factor. Higher values make distributions smoother.
        bw_inplace (bool): If True, enables in-place backward computation (simulated here for custom use cases).
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        torch.Tensor: The computed scaled cross-entropy loss.
    """
    if temp <= 0:
        raise ValueError("Temperature must be a positive value.")
    
    # Scale logits by temperature
    scaled_logits = logits / temp
    
    # Apply log softmax
    log_probs = F.log_softmax( scaled_logits, dim = -1 )
    
    # Compute negative log likelihood
    nll_loss = -log_probs[ torch.arange( logits.size(0) ), targets ]
    
    # Apply reduction
    if reduction == 'mean':
        loss = nll_loss.mean()
    elif reduction == 'sum':
        loss = nll_loss.sum()
    elif reduction == 'none':
        loss = nll_loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Choose from 'none', 'mean', or 'sum'.")
    
    if bw_inplace:
        loss.backward(create_graph=True)

    return loss

class DINOLoss(nn.Module):
    
    """
    Implementation of the loss function used in DINO (Distillation with No Labels).
    This loss function compares the outputs of a student network with those of a teacher network
    using cross-entropy while incorporating techniques like temperature scaling and centering.
    """

    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        """
        Initializes the DINOLoss module.
        
        Args:
            out_dim (int): Dimension of the output space (number of prototypes/classes).
            student_temp (float): Temperature used to scale student outputs for sharper distribution.
            center_momentum (float): Momentum for updating the center used in teacher output centering.
        """
        super().__init__()
        
        self.student_temp = student_temp  # Temperature for the student model.
        self.center_momentum = center_momentum  # Momentum for updating the center.
        
        # Center buffer, used to center the teacher output for stability.
        self.register_buffer( "center", torch.zeros( 1, out_dim ) )
        
        # Variables to manage distributed training and asynchronous operations.
        self.updated = True  # Flag to track if the center has been updated.
        self.reduce_handle = None  # Handle for asynchronous operations in distributed mode.
        self.len_teacher_output = None  # Tracks the number of samples in the teacher output batch.
        self.async_batch_center = None  # Accumulator for asynchronous updates to the center.

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        """
        Applies centering and sharpening (softmax) to the teacher output.
        
        Args:
            teacher_output (torch.Tensor): Output of the teacher network.
            teacher_temp (float): Temperature for sharpening the teacher's distribution.
        
        Returns:
            torch.Tensor: Sharpened and centered teacher output as a probability distribution.
        """
        # Ensure the center is up-to-date before applying it.
        self.apply_center_update()
        # Perform centering and temperature scaling, followed by softmax.
        return F.softmax( ( teacher_output - self.center ) / teacher_temp, dim = -1 )

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        """
        Normalizes the teacher's output using the Sinkhorn-Knopp algorithm.
        Ensures that the distribution sums to 1 for both rows (prototypes) and columns (samples).
        
        Args:
            teacher_output (torch.Tensor): Teacher output logits.
            teacher_temp (float): Temperature for scaling.
            n_iterations (int): Number of iterations to run the Sinkhorn-Knopp normalization.

        Returns:
            torch.Tensor: Row- and column-normalized teacher output (assignment matrix).
        """
        teacher_output = teacher_output.float()  # Ensure teacher output is in float precision.
        Q = torch.exp( teacher_output / teacher_temp ).t()  # Apply softmax-like operation and transpose.
        
        # Compute the number of samples (B) and prototypes (K).
        B = Q.shape[1] # Total number of samples across processes.
        K = Q.shape[0] # Number of prototypes (rows).
        
        # Normalize the matrix so that the sum of all elements equals 1.
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        # Iteratively normalize rows and columns using the Sinkhorn-Knopp algorithm.
        for it in range( n_iterations ):
            # Normalize rows: ensure total weight per prototype is 1/K.
            sum_of_rows = torch.sum( Q, dim = 1, keepdim = True )
            
            Q /= sum_of_rows
            Q /= K  # Normalize by the number of prototypes.

            # Normalize columns: ensure total weight per sample is 1/B.
            Q /= torch.sum( Q, dim = 0, keepdim = True )
            Q /= B  # Normalize by the total number of samples.

        # Scale the columns back to ensure each column sums to 1.
        Q *= B
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Computes the loss between the student and teacher outputs using cross-entropy.
        
        Args:
            student_output_list (list of torch.Tensor): List of student outputs (e.g., from multiple views).
            teacher_out_softmaxed_centered_list (list of torch.Tensor): List of centered teacher outputs.

        Returns:
            torch.Tensor: Total loss value.
        """
        total_loss = 0  # Initialize the loss accumulator.
        
        for s, t in zip( student_output_list, teacher_out_softmaxed_centered_list ):
            
            # Compute log-softmax of student output, scaling by the student temperature.
            lsm = F.log_softmax( s / self.student_temp, dim = -1 )
                            
            # Compute cross-entropy between teacher's probability distribution and student's log-softmax.
            loss = torch.sum( t * lsm, dim = -1 )
            total_loss -= loss.mean()  # Accumulate the negative mean loss.

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Updates the center buffer based on the teacher's output.
        
        Args:
            teacher_output (torch.Tensor): Teacher output logits.
        """
        self.reduce_center_update( teacher_output )

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        """
        Reduces (sums) the teacher's output across the batch and across processes (if distributed).
        Prepares for updating the center buffer.
        
        Args:
            teacher_output (torch.Tensor): Teacher output logits.
        """
        self.updated = False  # Mark the center as needing an update.
        self.len_teacher_output = len( teacher_output )  # Track the batch size of teacher output.
        self.async_batch_center = torch.sum( teacher_output, dim = 0, keepdim = True )  # Sum logits over the batch.

    @torch.no_grad()
    def apply_center_update(self):
        """
        Applies the updated center buffer after reducing teacher outputs.
        Incorporates momentum-based updates for stability.
        """
        if not self.updated:  # Only update if the center has been marked as needing an update.
            
            # Normalize the accumulated teacher outputs by the total number of samples.
            _t = self.async_batch_center / self.len_teacher_output

            # Update the center using momentum.
            self.center = self.center * self.center_momentum + _t * ( 1 - self.center_momentum )

            # Mark the center as updated.
            self.updated = True

def lossfunc(t, s, temp):
    return torch.sum( t * F.log_softmax( ( s / temp ), dim = -1 ), dim = -1 )

class iBOTPatchLoss(nn.Module):
    
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
    
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        self.apply_center_update()
        return F.softmax( ( teacher_patch_tokens.sub_( self.center ) ) / teacher_temp, dim = -1 )

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3):

        teacher_output = teacher_output.float()
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp( teacher_output / teacher_temp ).t()  # Q is K-by-B for consistency with notations from our paper
        # B = Q.shape[1] * world_size # number of samples to assign
        B = n_masked_patches_tensor        
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)        
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum( Q, dim = 1, keepdim = True )            
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum( Q, dim = 0, keepdim = True )
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = torch.sum( t * F.log_softmax( s / self.student_temp, dim = -1 ), dim = -1 )
        loss = torch.sum( loss * student_masks_flat.float(), dim = -1 ) / student_masks_flat.sum( dim = -1 ).clamp( min = 1.0 )
        return -loss.mean()

    def forward_masked(self, student_patch_tokens_masked, teacher_patch_tokens_masked,
                       student_masks_flat, n_masked_patches=None, masks_weight=None):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        loss = lossfunc( t, s, self.student_temp )
        if masks_weight is None:
            masks_weight = (
                ( 1 / student_masks_flat.sum( -1 ).clamp( min = 1.0 ) )
                .unsqueeze(-1)
                .expand_as( student_masks_flat )[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len( teacher_patch_tokens )
        self.async_batch_center = torch.sum( teacher_patch_tokens.mean(1), dim = 0, keepdim = True )
        
    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            _t = self.async_batch_center / self.len_teacher_patch_tokens
            self.center = self.center * self.center_momentum + _t * ( 1 - self.center_momentum )
            self.updated = True

class KoLeoLoss(nn.Module):

    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance( 2, eps = 1e-6 )

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm( x, x.t() )
        n = x.shape[0]
        dots.view(-1)[ :: ( n + 1 )].fill_( -1 )  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max( dots, dim = 1 )  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.amp.autocast( enabled = False, device_type = 'cuda' ):
            with torch.no_grad():
                I = self.pairwise_NNs_inner( F.normalize( student_output, eps = eps, p = 2, dim = -1 ) )  # noqa: E741
            distances = self.pdist( student_output, student_output[I] )  # BxD, BxD -> B
            loss = -torch.log( distances ).mean()
        return loss

class VicRegLoss(nn.Module):

    def __init__(self, sim_weight=25.0, var_weight=25.0, cov_weight=1.0, eps=1e-4):
        
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.eps = eps

    def forward(self, x, y):
        
        """
        Compute VICReg loss for batch of embeddings x and y.
        x and y: [B, D] tensors from two augmented views.
        Returns: scalar loss (variance, invariance, covariance terms combined).
        """
        B, D, H, W = x.shape
        x = F.adaptive_avg_pool2d( x, 1 ).view( B, D )
        y = F.adaptive_avg_pool2d( y, 1 ).view( B, D )

        # Invariance term (MSE between x and y)
        inv_loss = F.mse_loss( x, y )  # mean squared error over batch and dims
        
        # Variance term: ensure each dimension in x and y has std > 1 (or target = 1)
        # Compute standard deviation per dimension
        std_x = torch.sqrt( x.var( dim = 0 ) + self.eps )
        std_y = torch.sqrt( y.var( dim = 0 ) + self.eps )
        # Hinge loss: penalize if std < 1.0
        var_loss = torch.mean( F.relu( 1.0 - std_x ) ) + torch.mean( F.relu( 1.0 - std_y ) )
        
        # Covariance term: decorrelate features (off-diagonal covariance)
        x_cent = x - x.mean( dim = 0, keepdim = True )
        y_cent = y - y.mean( dim = 0, keepdim = True )
        # Compute covariance matrices (dimension D x D)
        cov_x = ( x_cent.T @ x_cent ) / ( B - 1 )
        cov_y = ( y_cent.T @ y_cent ) / ( B - 1 )
        # Zero-out diagonal for covariance matrices and square for penalty
        cov_offdiag_x = cov_x - torch.diag( torch.diag( cov_x ) )
        cov_offdiag_y = cov_y - torch.diag( torch.diag( cov_y ) )
        cov_loss = ( cov_offdiag_x**2 ).sum() / D + ( cov_offdiag_y**2 ).sum() / D
        
        # Combine weighted sum of all three terms
        loss = self.sim_weight * inv_loss + self.var_weight * var_loss + self.cov_weight * cov_loss
        
        return loss
    
def soft_maximum(x, y):
    return 0.5 * (x + y) + 0.5 * torch.abs(x - y)

def soft_minimum(x, y):
    return 0.5 * (x + y) - 0.5 * torch.abs(x - y)

def soft_binarize(x: torch.Tensor, threshold: float = 0.5, alpha: float = 10.0) -> torch.Tensor:
    """
    Approximate the binary step with a steep sigmoid:
        out = sigmoid( alpha * (x - threshold) )
    where alpha >> 1 makes the transition very sharp.
    """
    return torch.sigmoid( alpha * ( x - threshold ) )

def get_coordinates(bbox):

    # x_center, y_center, width, height = bbox
    x_min = ( bbox[...,0] - bbox[...,2] / 2 )
    y_min = ( bbox[...,1] - bbox[...,3] / 2 )
    x_max = ( bbox[...,0] + bbox[...,2] / 2 )
    y_max = ( bbox[...,1] + bbox[...,3] / 2 )

    return x_min, y_min, x_max, y_max

def get_unormalized_coordinates(bbox, w, h):

    # x_center, y_center, width, height = bbox
    x_min = ( bbox[...,0] - bbox[...,2] / 2 ) * w
    y_min = ( bbox[...,1] - bbox[...,3] / 2 ) * h
    x_max = ( bbox[...,0] + bbox[...,2] / 2 ) * w
    y_max = ( bbox[...,1] + bbox[...,3] / 2 ) * h

    return x_min, y_min, x_max, y_max

def compute_ciou_loss(pred, target, image_width, image_height):
    """
    Compute the Complete IoU (CIoU) loss between predicted and target boxes.

    Args:
        pred (torch.Tensor): Predicted boxes of shape [bs, ..., 4] (x, y, w, h).
        target (torch.Tensor): Target boxes of shape [bs, ..., 4] (x, y, w, h).

    Returns:
        torch.Tensor: CIoU loss tensor of shape [bs, ...].
    """

    # Convert normalized box coordinates to pixel coordinates.
    pred_x1, pred_y1, pred_x2, pred_y2 = get_unormalized_coordinates( pred, image_width, image_height )
    target_x1, target_y1, target_x2, target_y2 = get_unormalized_coordinates( target, image_width, image_height )

    # Intersection coordinates.
    inter_x1 = soft_maximum( pred_x1, target_x1 )
    inter_y1 = soft_maximum( pred_y1, target_y1 )
    inter_x2 = soft_minimum( pred_x2, target_x2 )
    inter_y2 = soft_minimum( pred_y2, target_y2 )

    # Intersection area.
    inter_w = F.relu( inter_x2 - inter_x1 ) # torch.clamp( inter_x2 - inter_x1, min = 0 )
    inter_h = F.relu( inter_y2 - inter_y1 ) # torch.clamp( inter_x2 - inter_x1, min = 0 )
    inter_area = inter_w * inter_h

    # Areas of predicted and target boxes.
    pred_area = ( pred_x2 - pred_x1 ) * ( pred_y2 - pred_y1 )
    target_area = ( target_x2 - target_x1 ) * ( target_y2 - target_y1 )

    # Union area.
    union_area = pred_area + target_area - inter_area

    # IoU.
    iou = inter_area / ( union_area + 1e-8 )
    iou_error = F.relu( 1.0 - iou )

    # Center coordinates.
    pred_center_x = ( pred_x1 + pred_x2 ) / 2
    pred_center_y = ( pred_y1 + pred_y2 ) / 2
    target_center_x = ( target_x1 + target_x2 ) / 2
    target_center_y = ( target_y1 + target_y2 ) / 2

    # Squared distance between centers.
    center_dist_sq = ( pred_center_x - target_center_x ).pow( 2 ) + ( pred_center_y - target_center_y ).pow( 2 )

    # Smallest enclosing box.
    enc_x1 = soft_minimum( pred_x1, target_x1 )
    enc_y1 = soft_minimum( pred_y1, target_y1 )
    enc_x2 = soft_maximum( pred_x2, target_x2 )
    enc_y2 = soft_maximum( pred_y2, target_y2 )
    enc_diag_sq = ( enc_x2 - enc_x1 ).pow( 2 ) + ( enc_y2 - enc_y1 ).pow( 2 )

    # Compute u term.
    u = center_dist_sq / ( enc_diag_sq + 1e-8 )

    # Aspect ratio consistency term.
    pred_w = pred_x2 - pred_x1
    pred_h = pred_y2 - pred_y1
    target_w = target_x2 - target_x1
    target_h = target_y2 - target_y1

    pred_ar = pred_w / ( pred_h + 1e-8 )
    target_ar = target_w / ( target_h + 1e-8 )
    v = 4 / ( torch.pi ** 2 ) * ( torch.atan( target_ar ) - torch.atan( pred_ar ) ).pow( 2 )

    # Alpha parameter.
    alpha = soft_binarize( iou, 0.5 ) * ( v / ( iou_error + v + 1e-8 ) )

    # CIoU loss.
    ciou = iou_error + u + alpha * v

    return ciou  # shape: [bs, num_tokens, num_objects]

def compute_gaussian_kl_loss(pred, target, image_width, image_height, alpha=1.0, beta=1.0):
    
    pred_x1, pred_y1, pred_x2, pred_y2 = get_unormalized_coordinates( pred, image_width, image_height )
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = get_unormalized_coordinates( target, image_width, image_height )

    # --- 2) Get (cx, cy, w_box, h_box) for predicted/target boxes in pixels.
    pred_cx = 0.5 * ( pred_x1 + pred_x2 )
    pred_cy = 0.5 * ( pred_y1 + pred_y2 )
    pred_w  = ( pred_x2 - pred_x1 ).clamp_min( 1e-6 )  # avoid zero or negative
    pred_h  = ( pred_y2 - pred_y1 ).clamp_min( 1e-6 )

    tgt_cx  = 0.5 * ( tgt_x1 + tgt_x2 )
    tgt_cy  = 0.5 * ( tgt_y1 + tgt_y2 )
    tgt_w   = ( tgt_x2 - tgt_x1 ).clamp_min( 1e-6 )
    tgt_h   = ( tgt_y2 - tgt_y1 ).clamp_min( 1e-6 )

    # --- 3) For each box, define the 4D mean vector and diagonal covariance:
    #         mean = (cx, cy, w, h)
    #         diag_sigma^2 = [ (alpha*w)^2, (alpha*h)^2, (beta*w)^2, (beta*h)^2 ]
    #     You can tweak alpha, beta or define something else. 
    #     We keep it consistent for demonstration.
    
    # Means (pred)
    mu_pred = torch.stack( [ pred_cx, pred_cy, pred_w, pred_h ], dim = -1 )  # shape [bs, ..., 4]
    # Diagonal variance
    var_pred_cx = ( alpha * pred_w ).pow(2)  # shape [bs, ...]
    var_pred_cy = ( alpha * pred_h ).pow(2)
    var_pred_w  = ( beta  * pred_w ).pow(2)
    var_pred_h  = ( beta  * pred_h ).pow(2)
    # Combine
    var_pred = torch.stack( [ var_pred_cx, var_pred_cy, var_pred_w, var_pred_h ], dim = -1 ).clamp_min( 1e-8 )

    # Means (target)
    mu_tgt = torch.stack( [ tgt_cx, tgt_cy, tgt_w, tgt_h ], dim = -1 )
    # Diagonal variance
    var_tgt_cx = ( alpha * tgt_w ).pow(2)
    var_tgt_cy = ( alpha * tgt_h ).pow(2)
    var_tgt_w  = ( beta  * tgt_w ).pow(2)
    var_tgt_h  = ( beta  * tgt_h ).pow(2)
    var_tgt = torch.stack( [ var_tgt_cx, var_tgt_cy, var_tgt_w, var_tgt_h ], dim = -1 ).clamp_min( 1e-8 )

    # --- 4) Define a small helper that computes KL(N1||N2) for diagonal 4D Gaussians
    #     KL(N1||N2) = 0.5 * [ sum_j( var1_j / var2_j + 
    #                                (mu2_j - mu1_j)^2 / var2_j - 1 + 
    #                                log(var2_j / var1_j) ) ]
    #
    # We'll then do symmetrical KL = 0.5*(KL(N1||N2) + KL(N2||N1)).

    def diag_gaussian_kl(mu1, var1, mu2, var2):
        # var1, var2 are diagonal (shape [..., 4])
        # mu1, mu2 are shape [..., 4]
        # Return shape [..., ]
        eps = 1e-9
        log_var_ratio = torch.log( var2 + eps ) - torch.log( var1 + eps )
        term1 = ( var1 / ( var2 + eps ) ).sum( dim = -1 )  # sum over 4 coords
        term2 = ( ( mu2 - mu1 ).pow(2) / ( var2 + eps ) ).sum( dim = -1 )
        term3 = -4  # dimension is 4 => sum of (-1) across j=1..4
        term4 = log_var_ratio.sum( dim = -1 )

        kl = 0.5 * ( term1 + term2 + term3 + term4 )
        return kl

    kl_pred_tgt = diag_gaussian_kl( mu_pred, var_pred, mu_tgt, var_tgt )  # KL(pred||tgt)
    kl_tgt_pred = diag_gaussian_kl( mu_tgt, var_tgt, mu_pred, var_pred )  # KL(tgt||pred)

    # symmetrical KL distance
    kl_sym = 0.5 * ( kl_pred_tgt + kl_tgt_pred )  # shape: [bs, ...]

    return kl_sym

def unormalized_l1_loss(pred, target, image_width, image_height):

    pred_x1, pred_y1, pred_x2, pred_y2 = get_unormalized_coordinates( pred, image_width, image_height )
    target_x1, target_y1, target_x2, target_y2 = get_unormalized_coordinates( target, image_width, image_height )

    pred_ = torch.stack( [ pred_x1, pred_y1, pred_x2, pred_y2 ], dim = -1 )
    target_ = torch.stack( [ target_x1, target_y1, target_x2, target_y2 ], dim = -1 )
    # Compute the L1 loss between the predicted and target boxes.
    loss = torch.abs( pred_ - target_ ).sum( dim = -1 )  # shape: [bs, num_tokens, num_objects]

    return loss

def normalized_l1_loss(pred, target):

    pred_x1, pred_y1, pred_x2, pred_y2 = get_coordinates( pred )
    target_x1, target_y1, target_x2, target_y2 = get_coordinates( target )

    pred_ = torch.stack( [ pred_x1, pred_y1, pred_x2, pred_y2 ], dim = -1 )
    target_ = torch.stack( [ target_x1, target_y1, target_x2, target_y2 ], dim = -1 )
    # Compute the L1 loss between the predicted and target boxes.
    loss = torch.abs( pred_ - target_ ).sum( dim = -1 )  # shape: [bs, num_tokens, num_objects]

    return loss

@torch.no_grad()
def bbox_target_to_classes(target_bbox, num_bins=100):
    """
    Convert continuous bounding box targets to discrete class indices.
    
    Args:
        target_bbox (torch.Tensor): Tensor of shape [B, T, 4] with values in [0, 1] representing 
                                    [top, left, width, height].
        num_bins (int): Number of discrete bins/classes for each coordinate.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Class indices for 
            top, left, width, height. Each tensor is of shape [B, T] with integer values in [0, num_bins-1].
    """
    # Multiply by num_bins and take the floor to obtain class index.
    target_classes = ( target_bbox * num_bins ).floor().long()
    # Clamp to ensure indices are in [0, num_bins-1]
    target_classes = target_classes.clamp( max = num_bins - 1)
    
    # Split into four components.
    cx = target_classes[..., 0]   # [B, T]
    cy = target_classes[..., 1]    # [B, T]
    width_cls = target_classes[..., 2]  # [B, T]
    height_cls = target_classes[..., 3] # [B, T]
    
    return cx, cy, width_cls, height_cls

def compute_bbox_classification_loss(pred_logits, target_bbox, num_bins=100):
    
    B, _ = pred_logits[0].shape  # assume all logits have the same shape

    # Convert the continuous bounding box target into class indices.
    cx_target, cy_target, width_target, height_target = bbox_target_to_classes( target_bbox, num_bins = num_bins )
    
    # Flatten the batch and query dimensions to shape [B*T, num_bins] for cross entropy.
    loss_cx = F.cross_entropy( pred_logits[0], cx_target )
    loss_cy = F.cross_entropy( pred_logits[1], cy_target )
    loss_width = F.cross_entropy( pred_logits[2], width_target )
    loss_height = F.cross_entropy( pred_logits[3], height_target )
    
    total_loss = loss_cx + loss_cy + loss_width + loss_height
    return total_loss, { "loss_cy": loss_cy.item(), "loss_cx": loss_cx.item(), "loss_width": loss_width.item(), "loss_height": loss_height.item() }

def bbox_iou(box1, box2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    
    b1_x1, b1_y1, b1_x2, b1_y2 = get_coordinates( box1 )
    b2_x1, b2_y1, b2_x2, b2_y2 = get_coordinates( box2 )
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = ( b1_x2.minimum( b2_x2 ) - b1_x1.maximum( b2_x1 ) ).clamp_(0) * (
        b1_y2.minimum( b2_y2 ) - b1_y1.maximum( b2_y1 )
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum( b2_x2 ) - b1_x1.minimum( b2_x1 )  # convex (smallest enclosing box) width
        ch = b1_y2.maximum( b2_y2 ) - b1_y1.minimum( b2_y1 )  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                ( b2_x1 + b2_x2 - b1_x1 - b1_x2 ).pow(2) + ( b2_y1 + b2_y2 - b1_y1 - b1_y2 ).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = ( 4 / math.pi**2 ) * ( ( w2 / h2 ).atan() - ( w1 / h1 ).atan() ).pow( 2 )
                with torch.no_grad():
                    alpha = v / ( v - iou + ( 1 + eps ) )
                return iou - ( rho2 / c2 + v * alpha )  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - ( c_area - union ) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

class FocalLoss(nn.Module):

    def __init__(self):

        super().__init__()

    @staticmethod
    def forward(pred, targets, gamma=1.5, alpha=0.25):

        bs, nc = pred.shape
        one_hot = torch.zeros( ( bs, nc + 1 ), dtype = torch.int64, device = targets.device )
        one_hot.scatter_( 1, targets.unsqueeze(-1), 1 )
        one_hot = one_hot[..., :-1].to( pred.dtype )  # [bs, nc]

        loss = F.binary_cross_entropy_with_logits( pred, one_hot, reduction = "none" )
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = one_hot * pred_prob + ( 1 - one_hot ) * ( 1 - pred_prob )
        modulating_factor = ( 1.0 - p_t ) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = one_hot * alpha + ( 1 - one_hot ) * ( 1 - alpha )
            loss *= alpha_factor
        return loss.mean(1).sum()

class VarifocalLoss(nn.Module):
    
    def __init__(self):

        super().__init__()

    @staticmethod
    def forward(pred_score, targets, gt_scores, alpha=0.75, gamma=2.0):

        bs, nq, nc = pred_score.shape
        one_hot = torch.zeros( ( bs, nq, nc + 1 ), dtype = torch.int64, device = targets.device )
        one_hot.scatter_( 2, targets.unsqueeze(-1), 1 )
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view( bs, nq, 1 ) * one_hot
        
        weight = alpha * pred_score.sigmoid().pow( gamma ) * ( 1 - one_hot ) + gt_scores * one_hot
        with torch.amp.autocast( enabled = False ):
            loss = (
                ( F.binary_cross_entropy_with_logits( pred_score.float(), gt_scores.float(), reduction = "none") * weight )
                .mean(1)
                .sum()
            )
        return loss

class HungarianLossComputation(nn.Module):

    def __init__(self, lambda_bbox=1.0, lambda_cls=1.0, image_width=640, image_height=480, num_classes=80):
        """
        Initialize the loss computation module.
        
        Args:
            lambda_bbox: Weight for the bounding box regression loss.
            lambda_cls: Weight for the classification loss.
            lambda_mask: Weight for the mask prediction loss.
            lambda_overlap: Weight for an optional overlap loss.
            image_width, image_height: Dimensions for scaling normalized coordinates.
            num_classes: Number of classes.
        """
        super(HungarianLossComputation, self).__init__()
        self.lambda_bbox = lambda_bbox
        self.lambda_cls = lambda_cls
        self.image_width = image_width
        self.image_height = image_height
        self.eps = 1e-8  # for numerical stability
        self.num_classes = num_classes

    def compute_pairwise_ciou_loss(self, pred, target):
        """
        Compute the Complete IoU (CIoU) loss between predicted and target boxes.

        Args:
            pred (torch.Tensor): Predicted boxes of shape [bs, num_tokens, 4] (x, y, w, h).
            target (torch.Tensor): Target boxes of shape [bs, num_objects, 4] (x, y, w, h).

        Returns:
            torch.Tensor: CIoU loss tensor of shape [bs, num_tokens, num_objects].
        """
        # Expand dimensions to enable pairwise computation:
        # pred: [bs, num_tokens, 1, 4] and target: [bs, 1, num_objects, 4]
        pred = pred.unsqueeze( 2 )      # [bs, num_tokens, 1, 4]
        target = target.unsqueeze( 1 )  # [bs, 1, num_objects, 4]
        return 1.0 - bbox_iou( pred, target, GIoU = True )

    def hungarian_matching(self, cost_matrix):
        """
        Perform Hungarian matching for each sample in the batch.

        Args:
            cost_matrix (torch.Tensor): Cost matrix of shape [bs, num_tokens, num_objects].

        Returns:
            Tuple: Two lists (one per batch element) containing predicted indices and target indices.
        """
        bs = cost_matrix.size(0)
        all_pred_indices = []
        all_target_indices = []
        cost_matrix_np = cost_matrix.cpu().detach().numpy()
        for b in range(bs):
            cost_b = cost_matrix_np[b]  # [num_tokens, num_objects]
            pred_idx, tgt_idx = linear_sum_assignment( cost_b )
            all_pred_indices.append( torch.as_tensor( pred_idx, dtype = torch.long, device = cost_matrix.device ) )
            all_target_indices.append( torch.as_tensor( tgt_idx, dtype = torch.long, device = cost_matrix.device ) )
        all_pred_indices = torch.stack( all_pred_indices, dim = 0 )
        all_target_indices = torch.stack( all_target_indices, dim = 0 )
        return all_pred_indices, all_target_indices

    def forward(self,
                pred_bboxes, target_bboxes,
                pred_classes, target_classes):
        """
        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape [bs, num_tokens, 4].
            target_bboxes (torch.Tensor): Target bounding boxes, shape [bs, num_objects, 4].
            pred_classes (torch.Tensor): Predicted class logits, shape [bs, num_tokens, num_classes].
            target_classes (torch.Tensor): Target class indices, shape [bs, num_objects].

        Returns:
            Tuple: Predicted indices and target indices per batch, based on the overall cost matrix.
        """
     
        # a. Pairwise Bounding Box L1 Loss.
        # pairwise_bbox_l1_loss = normalized_l1_loss( 
        #     pred_bboxes.unsqueeze(2), 
        #     target_bboxes.unsqueeze(1),
        # )  # [bs, num_tokens, num_objects]

        # b. Pairwise Classification Loss.
        # Expand target and prediction to align dimensions.
        target_classes_expanded = target_classes.unsqueeze(1).expand( -1, pred_classes.shape[1], -1 ) # [bs, num_tokens, num_objects]
        pred_log_probs = F.log_softmax( pred_classes, dim = -1 )  # [bs, num_tokens, num_classes]
        pairwise_cls_neg_log = -pred_log_probs.gather( dim = -1, index = target_classes_expanded ) # [bs, num_tokens, num_objects]

        # c. Pairwise CIoU Loss.
        pairwise_bbox_ciou_loss = self.compute_pairwise_ciou_loss( pred_bboxes, target_bboxes )

        # d. Combine the pairwise losses using lambda weights.
        cost_matrix = ( self.lambda_bbox * pairwise_bbox_ciou_loss
                        + self.lambda_cls * pairwise_cls_neg_log )
        
        # cost_matrix = ( self.lambda_bbox * pairwise_bbox_l1_loss
        #                 + self.lambda_cls * pairwise_cls_neg_log )

        # e. Perform Hungarian matching per batch sample.
        pred_indices, target_indices = self.hungarian_matching( cost_matrix )
        
        return pred_indices, target_indices



