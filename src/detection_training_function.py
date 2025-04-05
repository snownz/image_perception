import torch
import torch.nn.functional as F
import numpy as np

def empty_like(x):
    return (
        torch.empty_like( x, dtype = torch.float32) if isinstance( x, torch.Tensor ) else np.empty_like( x, dtype = np.float32 )
    )

def xywh2xyxy(x):
    y = empty_like( x )  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def xyxy2xywh(x):
    y = empty_like(x)  # faster than clone/copy
    y[..., 0] = ( x[..., 0] + x[..., 2] ) / 2  # x center
    y[..., 1] = ( x[..., 1] + x[..., 3] ) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

@torch.no_grad()
def generate_noisy_bboxes(
    gt_boxes, 
    gt_labels, 
    num_classes, 
    num_copies=5, 
    box_noise_scale=0.1, 
    cls_noise_ratio=0.5):
    """
    Generate noisy versions of ground-truth boxes and labels.
    
    Args:
        gt_boxes (Tensor): [B, N, 4] tensor in normalized coordinates (x, y, w, h).
        gt_labels (Tensor): [B, N] tensor with class labels.
        num_copies (int): Number of noisy copies per ground-truth box per type (positive/negative).
        box_noise_scale (float): Magnitude of noise applied to bounding boxes.
        cls_noise_ratio (float): Fraction of negative boxes to randomly change class label.
        
    Returns:
        dn_bbox (Tensor): [B, N, 2*num_copies, 4] tensor of noisy boxes,
                          with ordering: [s0pb0, s0pb1, ..., s0nb0, s0nb1, ...].
        dn_labels (Tensor): [B, N, 2*num_copies] tensor of labels, with negative copies
                            possibly perturbed.
    """

    B, N, _ = gt_boxes.shape  # [B, N, 4]
    # Create copies for positive and negative samples.
    dn_bbox_pos = gt_boxes.unsqueeze(2).expand( -1, -1, num_copies, -1 ).clone()  # [B, N, num_copies, 4]
    dn_bbox_neg = gt_boxes.unsqueeze(2).expand( -1, -1, num_copies, -1 ).clone()  # [B, N, num_copies, 4]

    def apply_noise(x, scale, add=0.0):
        # Convert from (x, y, w, h) to (xmin, ymin, xmax, ymax)
        known_bbox = xywh2xyxy( x )
        # Compute a noise range based on half the box size and the given scale.
        diff = ( x[..., 2:] * 0.5 ).repeat( 1, 1, 1, 2 ) * scale
        rand_sign = torch.randint_like( x, 0, 2 ) * 2.0 - 1.0  # random sign (-1 or 1)
        rand_part = torch.rand_like( x )
        rand_part += add
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clamp_( min = 0.0, max = 1.0 )
        # Convert back to (x, y, w, h) and apply inverse sigmoid.
        x = xyxy2xywh( known_bbox )
        # x = torch.logit( x, eps = 1e-6 )
        return x

    # Apply noise to bounding boxes.
    dn_bbox_pos = apply_noise( dn_bbox_pos, box_noise_scale )
    dn_bbox_neg = apply_noise( dn_bbox_neg, box_noise_scale, 2.0 )

    # Process class labels.
    dn_labels_pos = gt_labels.unsqueeze(2).expand( -1, -1, num_copies ).clone()  # positive labels stay unchanged
    dn_labels_neg = gt_labels.unsqueeze(2).expand( -1, -1, num_copies ).clone()  # negative labels to be noised

    if cls_noise_ratio > 0:
        # For negative samples, randomly change class labels based on the noise ratio.
        noise_mask = torch.rand_like( dn_labels_neg, dtype = torch.float ) < cls_noise_ratio
        # Assuming class labels range from 0 to num_classes-1.
        random_labels = torch.randint( 0, num_classes - 1, dn_labels_neg.shape, dtype = dn_labels_neg.dtype, device = dn_labels_neg.device )
        dn_labels_neg = torch.where( noise_mask, random_labels, dn_labels_neg )

    # Stack along a new dimension representing the positive/negative type.
    dn_bbox_grouped = torch.stack( [ dn_bbox_pos, dn_bbox_neg ], dim = 3 )   # shape: [B, N, num_copies, 2, 4]
    dn_labels_grouped = torch.stack( [ dn_labels_pos, dn_labels_neg ], dim = 3 )  # shape: [B, N, num_copies, 2]

    # Permute dimensions to bring groups to the front:
    dn_bbox_permuted = dn_bbox_grouped.permute( 0, 2, 3, 1, 4 )    # shape: [ B, num_copies, 2, N, 4 ]
    dn_labels_permuted = dn_labels_grouped.permute( 0, 2, 3, 1 )  # shape: [ B, num_copies, 2, N ]

    dn_bbox_reordered = dn_bbox_permuted.reshape( B, num_copies * 2, N, 4 ) # shape: [B, N, 2*num_copies, 4]
    dn_labels_reordered = dn_labels_permuted.reshape( B, num_copies * 2, N ) # shape: [B, N, 2*num_copies]

    dn_bbox_final = dn_bbox_reordered.flatten(1,2)   # shape: [B, 2*num_copies*N, 4]
    dn_labels_final = dn_labels_reordered.flatten(1) # shape: [B, 2*num_copies*N]
    #[s0[g0p,g0n], s1[g0p,g0n], s2[g0p,g0n], ...] => [ s0g0p, s1g0p, s0g0n, s1g0n, s0g1p, s1g1p, s1g1n, s2g1n, ...]

    return dn_bbox_final, dn_labels_final

def extract_pos_neg_cls(pred_cls, target_cls):
    """
    Extract positive and negative classification predictions.
    
    Args:
        pred_cls (Tensor): Extended classification predictions with shape [B, L, C],
                           where L = num_groups * 2 * D and C is the number of classes.
        target_cls (Tensor): Ground-truth class labels with shape [B, D] (or [B, D, 1]).
    
    Returns:
        pos_cls (Tensor): Positive classification predictions, reordered as [B, D * num_groups, C].
        neg_cls (Tensor): Negative classification predictions, reordered as [B, D * num_groups, C].
    """
    D = target_cls.shape[1]  # number of detections
    pred_grouped = torch.split( pred_cls, D, dim = 2 ) # even are positive, odd are negative
    pos_cls = torch.cat( [ pred_grouped[i] for i in range( 0, len( pred_grouped ), 2 ) ], dim = 2 )  # shape: [B, num_groups, D, C]
    neg_cls = torch.cat( [ pred_grouped[i] for i in range( 1, len( pred_grouped ), 2 ) ], dim = 2 )  # shape: [B, num_groups, D, C]
    return pos_cls, neg_cls

def extract_pos_neg_boxes(pred_boxes, target_boxes):
    """
    Extract positive and negative bounding box predictions.
    
    Args:
        pred_boxes (Tensor): Extended prediction boxes with shape [B, L, 4],
                             where L = num_groups * 2 * D.
        target_boxes (Tensor): Ground-truth boxes with shape [B, D, 4].
    
    Returns:
        pos_boxes (Tensor): Positive predictions, reordered as [B, D * num_groups, 4].
        neg_boxes (Tensor): Negative predictions, reordered as [B, D * num_groups, 4].
    """
    D = target_boxes.shape[1]  # number of detections (per image)
    pred_grouped = torch.split( pred_boxes, D, dim = 2 ) # even are positive, odd are negative
    neg_boxes = torch.cat( [ pred_grouped[i] for i in range( 1, len( pred_grouped ), 2 ) ], dim = 2 )  # shape: [B, num_groups, D, 4]
    pos_boxes = torch.cat( [ pred_grouped[i] for i in range( 0, len( pred_grouped ), 2 ) ], dim = 2 )  # shape: [B, num_groups, D, 4]
    return pos_boxes, neg_boxes

@torch.no_grad()
def split_bboxes_by_area_left_aligned(gt_bboxes, labels, small_thresh=0.2, medium_thresh=0.6):
    """
    Split bounding boxes by area into three left-aligned tensors:
      - small: area < small_thresh
      - medium: small_thresh <= area < medium_thresh
      - large: area >= medium_thresh

    Args:
      gt_bboxes: [B, N, 4] in (x, y, w, h), normalized to [0..1].
      labels:    [B, N] class or object labels corresponding to those boxes
      small_thresh: upper bound for 'small' area (default=0.2)
      medium_thresh: upper bound for 'medium' area (default=0.6)

    Returns:
      (bboxes_small, bboxes_medium, bboxes_large), (labels_small, labels_medium, labels_large), (mask_small, mask_medium, mask_large)
        Each bboxes_* is [B, N, 4], each labels_* is [B, N], left-aligned:
          [obj, obj, obj, 0, 0, ...]
        The masks are boolean: [B, N].
    """
    B, N, _ = gt_bboxes.shape
    
    # 1) Compute area = w * h
    widths = gt_bboxes[..., 2]
    heights = gt_bboxes[..., 3]
    areas = widths * heights  # shape [B, N]

    # 2) Boolean masks for each category
    mask_small  = ( areas > 0 ) & ( areas < small_thresh )
    mask_medium = ( areas >= small_thresh ) & ( areas < medium_thresh )
    mask_large  = ( areas >= medium_thresh )

    # 3) Prepare output tensors
    bboxes_small  = torch.zeros_like( gt_bboxes )  # [B, N, 4]
    bboxes_medium = torch.zeros_like( gt_bboxes )  
    bboxes_large  = torch.zeros_like( gt_bboxes )

    labels_small  = torch.zeros_like( labels )  # [B, N]
    labels_medium = torch.zeros_like( labels )
    labels_large  = torch.zeros_like( labels )

    valid_small  = torch.zeros_like( labels )
    valid_medium = torch.zeros_like( labels )
    valid_large  = torch.zeros_like( labels )

    # 4) For each image in the batch, gather boxes for each category
    for i in range(B):
        
        # a) Small
        idx_small = torch.nonzero( mask_small[i], as_tuple = True)[0]  # shape [num_small]
        k_small = idx_small.shape[0]
        bboxes_small[i, :k_small, :] = gt_bboxes[i, idx_small, :]
        labels_small[i, :k_small]    = labels[i, idx_small]
        valid_small[i, :k_small]     = 1

        # b) Medium
        idx_medium = torch.nonzero( mask_medium[i], as_tuple = True )[0]
        k_medium = idx_medium.shape[0]
        bboxes_medium[i, :k_medium, :] = gt_bboxes[i, idx_medium, :]
        labels_medium[i, :k_medium]    = labels[i, idx_medium]
        valid_medium[i, :k_medium]     = 1

        # c) Large
        idx_large = torch.nonzero( mask_large[i], as_tuple = True )[0]
        k_large = idx_large.shape[0]
        bboxes_large[i, :k_large, :] = gt_bboxes[i, idx_large, :]
        labels_large[i, :k_large]    = labels[i, idx_large]
        valid_large[i, :k_large]     = 1

    return (
        ( bboxes_small, bboxes_medium, bboxes_large ),
        ( labels_small, labels_medium, labels_large ),
        ( valid_small, valid_medium, valid_large ),
    )