import torch
import torch.nn.functional as F
import numpy as np
import cv2
from flask import Flask, render_template, jsonify, request
from src.models import ObjectDetectionModel
from dataclasses import dataclass, field
from torch import nn
import random
import traceback
import math
import os
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Label mapping and random colors
int_to_label = {
    0: "person",
    1: "birds",
    2: "parking meter",
    3: "stop sign",
    4: "street sign",
    5: "fire hydrant",
    6: "traffic light",
    7: "motorcycle",
    8: "bicycle",
    9: "LMVs",
    10: "HMVs",
    11: "animals",
    12: "poles",
    13: "barricades",
    14: "traffic cones",
    15: "mailboxes",
    16: "stones",
    17: "small walls",
    18: "bins",
    19: "furniture",
    20: "pot plant",
    21: "sign boards",
    22: "boxes",
    23: "trees",
}

# Generate colors for labels and detectors - set random seed for consistency
random.seed(42)
det_to_color = { i: ( random.randint( 0, 255 ), random.randint( 0, 255 ), random.randint( 0, 255 ) ) for i in range( 300 ) }

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
    nc: int = 24
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
    name: str = "detection_v5_small"
    backbone_name: str = "encoder_v5"
    compute_precision: ComputePrecision = field(default_factory=ComputePrecision)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    hungarian_loss: HungarianLoss = field(default_factory=HungarianLoss)
    detection: Detection = field(default_factory=Detection)

def resize_image(image_tensor, size):
    return F.interpolate(image_tensor.unsqueeze(0), size=(size, size),
                         mode='bilinear', align_corners=False).squeeze(0)
                         
def draw_star(image, x, y, size, color, thickness=2):
    """Draw a star on the image at position (x,y) with given size and color"""
    # Calculate star points
    outer_radius = size
    inner_radius = size // 2
    points = []
    
    for i in range(10):
        # Use alternating outer and inner radius
        radius = outer_radius if i % 2 == 0 else inner_radius
        # Calculate angle (5 points = 10 vertices, 2π/10 radians per step)
        angle = math.pi/5 * i
        # Calculate point coordinates
        px = int(x + radius * math.sin(angle))
        py = int(y - radius * math.cos(angle))
        points.append((px, py))
    
    # Convert points to numpy array
    points = np.array(points, dtype=np.int32)
    
    # Draw the star as a filled polygon
    cv2.polylines(image, [points], True, color, thickness)
    
    return image

# Global variables
hook_outputs = {}
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
original_img_size = ( 640, 640 ) # Default size for display scaling
detection_results = {
    'boxes': None,
    'scores': None,
    'labels': None,
    'detectors': None,
    'indices': None
}

# Helper function to track occurrences for detections
def occurrence_indices(strings):
    counts = {}
    indices = []
    for s in strings:
        # Get the current count (default is 0)
        count = counts.get(s, 0)
        indices.append(count)
        # Update the count for this string
        counts[s] = count + 1
    return indices

# Hook function
def hook_fn(capture):
    def execute_hook(module, input, output):
        global hook_outputs
        hook_outputs[capture] = {
            'input': input,
            'output': output
        }
    return execute_hook

def load_model():
    global model
    cfg = Config()
    model = ObjectDetectionModel(cfg, device)
    model.to(device)
    model.eval()
    
    # Register hooks for different feature layers
    model.backbone.backbone.net.model.layers[0].register_forward_hook(hook_fn('feature_maps'))
    model.backbone.backbone.net.model.layers[25].register_forward_hook(hook_fn('features_small'))
    model.backbone.backbone.net.model.layers[28].register_forward_hook(hook_fn('features_medium'))
    model.backbone.backbone.net.model.layers[31].register_forward_hook(hook_fn('features_large'))
    model.detection.enc_score_head.register_forward_hook(hook_fn('proposal_queries')) # For tokens/proposals
    for i, l in enumerate( model.detection.decoder.layers ):
        l.self_attn.register_forward_hook(hook_fn(f'decoder_queries_attention_layer_{i}'))
    
# Global variable to store the current image path
current_image_path = 'test_images/8.png'

def process_image(image_path=None):
    global hook_outputs, model, original_img_size, detection_results, current_image_path
    
    # Use provided image path or the current global one
    if image_path is None:
        image_path = current_image_path
    else:
        current_image_path = image_path  # Update the global image path
    
    # Clear previous hooks output
    hook_outputs = {}
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_img_size = (image.shape[1], image.shape[0])  # Store original size for display scaling
    
    torch_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    torch_image = torch_image.to(device)
    torch_image = resize_image(torch_image, 640)
    
    # Run inference
    with torch.no_grad():
        boxes, scores, labels, detectors = model.predict(torch_image, stride_slices=32, confidence_threshold=0.3, iou_threshold=0.3)
        
    # Store detection results
    detection_results['boxes'] = boxes.cpu().numpy()
    detection_results['scores'] = scores.cpu().numpy()
    detection_results['labels'] = labels.cpu().numpy()
    detection_results['detectors'] = detectors.cpu().numpy()
    detection_results['indices'] = occurrence_indices(labels.cpu().numpy())

    indices = occurrence_indices(detection_results['labels'])
    detection_results['labels'] = [f"{int_to_label[l]}_{i}" for l, i in zip(detection_results['labels'], indices)]
    
    # Return the processed image for display
    return image

def extract_features_data():
    global hook_outputs, original_img_size
    
    # Mapping of module keys to readable names
    module_display_names = {
        'feature_maps': 'Feature Maps (Basic)',
        'features_small': 'Small Objects Detection Feat',
        'features_medium': 'Medium Objects Detection Feat',
        'features_large': 'Large Objects Detection Feat'
    }
    
    # Get all hook keys for feature maps
    hook_keys = ['feature_maps', 'features_small', 'features_medium', 'features_large']
    
    features_data = {}
    
    for key in hook_keys:
        if key in hook_outputs:
            # Extract and process features from this layer
            features = hook_outputs[key]['output'][0]
            
            features_data[key] = {
                'num_channels': features.shape[0],
                'height': features.shape[1],
                'width': features.shape[2],
                'display_name': module_display_names.get(key, key)
            }
    
    return features_data, hook_keys

def draw_detections_on_image(image):
    global detection_results, int_to_label

    labe_to_detector = { l:d for l, d in zip( detection_results['labels'], detection_results['detectors'] ) }
    int_to_color = { l: det_to_color[labe_to_detector[l]] for l in detection_results['labels'] }
    
    if detection_results['boxes'] is None:
        return image
    
    # Make a copy of the image to draw on
    image_result = image.copy()
    image_height, image_width = image_result.shape[0:2]
    
    # Draw all detection boxes using OpenCV instead of PIL
    for d in range(detection_results['boxes'].shape[0]):
        bbox = detection_results['boxes'][d]
        score = detection_results['scores'][d]
        label = detection_results['labels'][d]
        detector = detection_results['detectors'][d]
        
        # Convert from center format to corner format
        x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)
        
        # Get metadata for drawing
        label_name = label
        score_percentage = score.item() * 100
        detector_number = detector.item()
        box_color = int_to_color[label]
        
        # Draw rectangle
        cv2.rectangle(image_result, (x_min, y_min), (x_max, y_max), box_color, 2)
        
        # Draw text
        text = f"{label_name} {score_percentage:.1f}% #{detector_number}"
        cv2.putText(image_result, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 2)
        cv2.putText(image_result, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1)
    
    return image_result

def process_detections_tokens(shapes):
    global hook_outputs
    
    if 'proposal_queries' not in hook_outputs:
        return [], [], []
    
    # Get the outputs from the hook
    enc_outputs_scores = hook_outputs['proposal_queries']['output'].to('cpu')  # [bs, h*w, num_classes]
    
    # Get shape information
    bs, hw_total, num_classes = enc_outputs_scores.shape
    
    # Apply sigmoid to the scores
    enc_outputs_scores = enc_outputs_scores.sigmoid()
    
    # Get the max scores and labels
    scores, _ = enc_outputs_scores.max(-1)
    
    # get top-k tokens and values from the all 3 feature maps
    topk_values, topk_indices = torch.topk(scores, 300, dim=1)
    topk_indices = topk_indices[0]  # shape: (300,)
    topk_values = topk_values[0]    # shape: (300,)
    indices_idx = torch.arange(0, 300)  # shape: (300,)
    
    level0_end = shapes[0][0] * shapes[0][1]
    level1_end = level0_end + shapes[1][0] * shapes[1][1]
    level2_end = level1_end + shapes[2][0] * shapes[2][1]  # Should equal hw_total
    
    # Isolate tokens for level 0: indices in [0, level0_end)
    mask0 = (topk_indices < level0_end)
    top_indices_feat_0 = topk_indices[mask0]
    top_values_feat_0 = topk_values[mask0]
    indices_idx_feat_0 = indices_idx[mask0]
    
    # Isolate tokens for level 1: indices in [level0_end, level1_end)
    mask1 = (topk_indices >= level0_end) & (topk_indices < level1_end)
    top_indices_feat_1 = topk_indices[mask1] - level0_end
    top_values_feat_1 = topk_values[mask1]
    indices_idx_feat_1 = indices_idx[mask1]
    
    # Isolate tokens for level 2: indices in [level1_end, level2_end)
    mask2 = (topk_indices >= level1_end) & (topk_indices < level2_end)
    top_indices_feat_2 = topk_indices[mask2] - level1_end  # local index if desired
    top_values_feat_2 = topk_values[mask2]
    indices_idx_feat_2 = indices_idx[mask2]
    
    all_top_indices = [top_indices_feat_0, top_indices_feat_1, top_indices_feat_2]
    all_top_values = [top_values_feat_0, top_values_feat_1, top_values_feat_2]
    all_top_indices_idx = [indices_idx_feat_0, indices_idx_feat_1, indices_idx_feat_2]
    
    return all_top_indices, all_top_values, all_top_indices_idx

def get_proposals_heatmap_overlay(image):

    global detection_results, int_to_label, det_to_color
    
    if 'proposal_queries' not in hook_outputs or detection_results['labels'] is None:
        return image
    
    # Create the legend mapping for detectors
    labels = detection_results['labels']
    indices = detection_results['indices']
    detectors = detection_results['detectors']
    
    legend = {d: l for l, i, d in zip(labels, indices, detectors)}
    
    # Define scales for multi-scale detection
    shapes = [(80, 80), (40, 40), (20, 20)]
    
    # Process detection tokens to get indices, values, and indices positions for each scale
    all_top_indices, all_top_values, all_top_indices_idx = process_detections_tokens(shapes)
    
    original_h, original_w = image.shape[0:2]
    heatmap_overlay = np.zeros((original_h, original_w), dtype=np.float32)
    all_detector_coords = []
    
    # Loop through each scale and update the heatmap overlay
    for i, ((h, w), s_indices, values, dxs) in enumerate(zip(shapes, all_top_indices, all_top_values, all_top_indices_idx)):
        if len(s_indices) == 0:  # Skip if no indices for this scale
            continue
            
        scale_y, scale_x = original_h / h, original_w / w
        circle_radius = max(1, int(min(scale_y, scale_x) // 2))
        for idx, val, dx in zip(s_indices, values, dxs):
            idx = idx.item()
            feat_y, feat_x = idx // w, idx % w
            
            orig_y = int((feat_y + 0.5) * scale_y)
            orig_x = int((feat_x + 0.5) * scale_x)
            
            orig_y = min(max(0, orig_y), original_h - 1)
            orig_x = min(max(0, orig_x), original_w - 1)
            
            score_val = val.item()
            
            cv2.circle(heatmap_overlay, (orig_x, orig_y), circle_radius, score_val, -1)
            
            if dx.item() in detectors:
                all_detector_coords.append((orig_x, orig_y, circle_radius * 3, dx.item()))
    
    # Normalize the heatmap overlay
    if heatmap_overlay.max() > 0:
        heatmap_overlay = heatmap_overlay / heatmap_overlay.max()
    
    # Create the colored heatmap using the COLORMAP_HOT
    heatmap_color = cv2.applyColorMap((heatmap_overlay * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend the original image with the heatmap overlay
    image_rgb = image.copy()
    alpha = 0.7
    blended = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_color, alpha, 0)
    
    # Draw detector circles and text labels on the final image using OpenCV
    for (cx, cy, radius, n) in all_detector_coords:
        # Draw circle (using the color from det_to_color)
        cv2.circle(blended, (cx, cy), radius, det_to_color[n], thickness=2)
        # Draw text label near the circle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(blended, legend[n], (cx + 10, cy - 15), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    
    return blended

@app.route('/')
def index():
    return render_template('features.html')

@app.route('/api/image')
def get_image():
    image = process_image()
    # Draw detections on a copy of the image
    image_with_detections = draw_detections_on_image(image.copy())
    
    # Create heatmap overlay on another copy
    heatmap_overlay = get_proposals_heatmap_overlay(image.copy())
    
    # Convert to base64 for sending to frontend
    _, buffer_original = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    _, buffer_detections = cv2.imencode('.png', cv2.cvtColor(image_with_detections, cv2.COLOR_RGB2BGR))
    _, buffer_heatmap = cv2.imencode('.png', cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
    
    # Prepare detection results for the frontend
    detection_data = {}
    if detection_results['boxes'] is not None:
        detection_data = {
            'boxes': detection_results['boxes'].tolist() if hasattr(detection_results['boxes'], 'tolist') else detection_results['boxes'],
            'scores': [float(s.item()) if hasattr(s, 'item') else float(s) for s in detection_results['scores']],
            'labels': detection_results['labels'],
            'detectors': detection_results['detectors'].tolist() if hasattr(detection_results['detectors'], 'tolist') else detection_results['detectors'],
            'indices': detection_results['indices']
        }
    
    return jsonify({
        'original': buffer_original.tobytes().hex(),
        'detections': buffer_detections.tobytes().hex(),
        'heatmap': buffer_heatmap.tobytes().hex(),
        'width': image.shape[1],
        'height': image.shape[0],
        'detection_results': detection_data
    })

@app.route('/api/label_mapping')
def get_label_mapping():
    global int_to_label, det_to_color
    
    # Convert color tuples to lists for JSON serialization
    serializable_colors = {str(k): list(v) for k, v in det_to_color.items()}
    
    return jsonify({
        'int_to_label': int_to_label,
        'det_to_color': serializable_colors
    })

@app.route('/api/features')
def get_features():
    features_data, hook_keys = extract_features_data()
    
    # Get all attention layer keys
    attention_layer_keys = [key for key in hook_outputs.keys() if key.startswith('decoder_queries_attention_layer_')]
    
    return jsonify({
        'features_data': features_data,
        'module_keys': hook_keys,
        'attention_layer_keys': attention_layer_keys
    })

@app.route('/api/channel/<module>/<int:channel_idx>')
def get_channel(module, channel_idx):
    global hook_outputs, original_img_size
    
    if module not in hook_outputs:
        return jsonify({'error': f'Module {module} not found'})
    
    features = hook_outputs[module]['output'][0]
    
    if channel_idx >= features.shape[0]:
        return jsonify({'error': 'Channel index out of range'})
    
    # Get the specific channel
    channel = features[channel_idx].cpu().numpy()
    
    # Resize the feature map to match original image dimensions for display
    # First normalize to 0-255
    channel_norm = ((channel - channel.min()) / (channel.max() - channel.min() + 1e-8) * 255).astype(np.uint8)
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(channel_norm, cv2.COLORMAP_HOT)
    
    # Resize to original image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_img_size[0], original_img_size[1]), 
                                interpolation=cv2.INTER_LINEAR)
    
    # Convert to base64 for sending to frontend
    _, buffer = cv2.imencode('.png', heatmap_resized)
    channel_b64 = buffer.tobytes()
    
    return jsonify({
        'channel_data': channel_b64.hex(),
        'min_val': float(channel.min()),
        'max_val': float(channel.max()),
        'mean_val': float(channel.mean()),
        'shape': {
            'height': int(channel.shape[0]),
            'width': int(channel.shape[1])
        }
    })

@app.route('/api/attention/<layer_key>/<int:query_idx>')
def get_attention_weights(layer_key, query_idx):
    global hook_outputs, detection_results
    
    if layer_key not in hook_outputs:
        return jsonify({'error': f'Attention layer {layer_key} not found'})
    
    # Get attention weights
    attention_output = hook_outputs[layer_key]['output']
    
    # For self-attention in transformer decoder, the attention matrix is [batch, num_heads, num_queries, num_queries]
    # We take the first batch and average over heads
    if isinstance(attention_output, tuple):
        # Some attention modules return (attn_output, attn_weights)
        attention_weights = attention_output[1][0].cpu().numpy()
    else:
        # Others might return just the attention tensor directly
        attention_weights = attention_output[0].cpu().numpy()
    
    # Average over heads if there are multiple
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    
    # Extract weights for the specified query
    query_attention = attention_weights[query_idx].tolist()
    
    # Extract associated query information from detection_results
    query_info = {}
    
    # Find this query in the detection results
    if detection_results['detectors'] is not None:
        detector_list = detection_results['detectors'].tolist()
        try:
            detection_index = detector_list.index(query_idx)
            if detection_index >= 0:
                query_info = {
                    'is_detection': True,
                    'label': detection_results['labels'][detection_index] if isinstance(detection_results['labels'], list) else detection_results['labels'][detection_index].item(),
                    'score': float(detection_results['scores'][detection_index].item()) if hasattr(detection_results['scores'][detection_index], 'item') else float(detection_results['scores'][detection_index])
                }
        except (ValueError, IndexError):
            # Query not found in detection results
            query_info = {'is_detection': False}
    
    return jsonify({
        'query_idx': query_idx,
        'attention_weights': query_attention,
        'query_info': query_info
    })

@app.route('/api/highlighted_heatmap/<layer_key>/<int:query_idx>')
def get_highlighted_heatmap(layer_key, query_idx):
    global hook_outputs, detection_results
    
    # Start with the base heatmap
    image = process_image()
    base_heatmap = get_proposals_heatmap_overlay(image.copy())
    
    # Get attention weights for the selected query
    if layer_key in hook_outputs:
        try:
            # Get attention weights
            attention_output = hook_outputs[layer_key]['output']
            
            # For self-attention in transformer decoder, get attention weights
            if isinstance(attention_output, tuple):
                attention_weights = attention_output[1][0].cpu().numpy()
            else:
                attention_weights = attention_output[0].cpu().numpy()
            
            # Average over heads if there are multiple
            if len(attention_weights.shape) > 2:
                attention_weights = attention_weights.mean(axis=0)
            
            # Get the weights for the selected query
            query_attention = attention_weights[query_idx]

            # Get the top 95% percentile of attention weights
            threshold = np.percentile( query_attention, 95 )
            query_attention = np.where( query_attention > threshold, query_attention, -1e8 )

            # Find the top 50 most attended queries
            top_indices = np.argsort(query_attention)[-10:]
            
            # Get detector coordinates for highlighting
            all_top_indices, all_top_values, all_top_indices_idx = process_detections_tokens([(80, 80), (40, 40), (20, 20)])
            
            original_h, original_w = base_heatmap.shape[0:2]
            all_detector_coords = {}
            
            # Create a mapping of detector ID to coordinates
            # Loop through each scale and identify query locations
            for i, ((h, w), s_indices, values, dxs) in enumerate(zip([(80, 80), (40, 40), (20, 20)], all_top_indices, all_top_values, all_top_indices_idx)):
                if len(s_indices) == 0:  # Skip if no indices for this scale
                    continue
                    
                scale_y, scale_x = original_h / h, original_w / w
                circle_radius = max(1, int(min(scale_y, scale_x) // 2))
                
                for idx, val, dx in zip(s_indices, values, dxs):
                    idx = idx.item()
                    feat_y, feat_x = idx // w, idx % w
                    
                    orig_y = int((feat_y + 0.5) * scale_y)
                    orig_x = int((feat_x + 0.5) * scale_x)
                    
                    orig_y = min(max(0, orig_y), original_h - 1)
                    orig_x = min(max(0, orig_x), original_w - 1)
                    
                    # Store coordinates by detector id
                    all_detector_coords[dx.item()] = (orig_x, orig_y, circle_radius * 3)
            
            # Draw stars on the heatmap for the top attended queries
            for idx in top_indices:
                if idx in all_detector_coords:
                    x, y, radius = all_detector_coords[idx]
                    # Draw yellow star for top 50 connected queries
                    draw_star(base_heatmap, x, y, radius, (0, 255, 255), thickness=2)
            
            # Highlight the selected query with a white star if it exists in our coordinates
            if query_idx in all_detector_coords:
                x, y, radius = all_detector_coords[query_idx]
                # Draw white star for the selected query
                draw_star(base_heatmap, x, y, radius, (255, 255, 255), thickness=3)
            
            # Convert to base64 for sending to frontend
            _, buffer = cv2.imencode('.png', cv2.cvtColor(base_heatmap, cv2.COLOR_RGB2BGR))
            return jsonify({
                'heatmap': buffer.tobytes().hex()
            })
            
        except Exception as e:
            print(f"Error generating highlighted heatmap: {e}")
            traceback.print_exc()
    
    # If anything fails, return the original heatmap
    _, buffer = cv2.imencode('.png', cv2.cvtColor(base_heatmap, cv2.COLOR_RGB2BGR))
    return jsonify({
        'heatmap': buffer.tobytes().hex()
    })

# Create a upload folder
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'nn_visualizer_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    # If user does not select file, browser might submit an empty file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Secure the filename and save it
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the uploaded image
            process_image(filepath)
            return jsonify({'success': True, 'message': 'Image processed successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5001)