import os
import numpy as np
import json
import cv2
from flask import jsonify
import colorsys
from collections import OrderedDict
import random
from tqdm import tqdm

dataset_folder = 'dataset_detection'

def mask_to_fixed_points(mask, num_points=100):
    """
    Convert a 2D boolean/uint8 mask into a list of 'num_points' (x, y) points
    describing its contour.
    """
    # Ensure mask is uint8
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)

    # Find external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    # Choose the largest contour (by area)
    contour = max(contours, key=cv2.contourArea)
    
    # 'contour' has shape (N, 1, 2). Squeeze to (N, 2)
    contour = contour.squeeze(axis=1)

    # Find the point closest to the origin (0,0)
    # and rotate the contour so that point is first.
    distances = np.sum(contour**2, axis=1)  # squared distance to (0, 0) for each point
    closest_idx = np.argmin(distances)
    # Rotate so that 'closest_idx' becomes the first element
    contour = np.roll(contour, -closest_idx, axis=0)
    
    # Ensure consistent orientation (e.g. counter-clockwise).
    # If it's clockwise, reverse it.
    if cv2.contourArea(contour) < 0:
        contour = contour[::-1]
    
    # Compute the total arc length of the contour
    contour_closed = np.vstack([contour, contour[0]])
    
    # Compute cumulative arc length
    lengths = [0.0]
    arc_length = 0.0
    for i in range(1, len(contour_closed)):
        arc_length += np.linalg.norm(contour_closed[i] - contour_closed[i-1])
        lengths.append(arc_length)
    
    if arc_length == 0:
        # edge case: single point contour
        return [tuple(contour_closed[0])] * num_points

    # Uniformly sample 'num_points' along the contour
    step = arc_length / num_points
    sample_distances = [i * step for i in range(num_points)]
    
    sampled_points = []
    idx = 0  # index to track which segment we are in
    for d in sample_distances:
        # If we exceed the last segment, wrap around
        d_mod = d % arc_length  
        # Advance idx until we find the segment containing d_mod
        while not (lengths[idx] <= d_mod < lengths[idx+1]):
            idx += 1
            if idx >= len(lengths) - 1:
                # wrap around if needed
                idx = 0
                d_mod = d_mod % arc_length

        seg_length = lengths[idx+1] - lengths[idx]
        if seg_length == 0:
            # edge case: two consecutive identical points
            pt = contour_closed[idx]
        else:
            # Linear interpolation along the segment
            ratio = (d_mod - lengths[idx]) / seg_length
            pt = contour_closed[idx] + ratio * (contour_closed[idx+1] - contour_closed[idx])
        sampled_points.append(tuple(pt))

    return sampled_points

def normalize_mask_points(points, bbox, image_size):
    """
    Normalize mask points relative to the bounding box dimensions.
    """
    image_width, image_height = image_size  # Get image dimensions

    bbox_width = int(bbox[2] * image_width) + 1
    bbox_height = int(bbox[3] * image_height) + 1

    # Normalize points relative to the bounding box
    normalized_points = [
        (point[0] / bbox_width, point[1] / bbox_height) 
        for point in points
    ]

    return normalized_points

def parse_annotation_line(line, image_width, image_height):
    """
    Parses an annotation line and computes the bounding box from the polygon.
    """
    # Split the line into components
    parts = line.strip().split(' ')
    
    if len(parts) < 3:
        raise ValueError("Annotation line does not contain enough elements.")
    
    # Extract the class label
    label = int(parts[0])
    
    # Extract the mask (polygon) coordinates
    mask_coords_norm = list(map(float, parts[1:]))
    
    if len(mask_coords_norm) % 2 != 0:
        raise ValueError("Number of mask coordinates is not even. Each point requires an x and y value.")
    
    # Pair the normalized coordinates and denormalize them
    polygon = []
    for i in range(0, len(mask_coords_norm), 2):
        x_norm, y_norm = mask_coords_norm[i], mask_coords_norm[i+1]
        x = int(x_norm * image_width)
        y = int(y_norm * image_height)
        polygon.append((x, y))
    
    if not polygon:
        raise ValueError("No valid polygon points found.")
    
    # Convert to NumPy array suitable for OpenCV
    polygon_np = np.array([polygon], dtype=np.int32)
    
    # Compute bounding box from polygon points
    x_coords = polygon_np[:, :, 0].flatten()
    y_coords = polygon_np[:, :, 1].flatten()
    
    x_min = max(int(np.min(x_coords)) - 1, 0)
    y_min = max(int(np.min(y_coords)) - 1, 0)
    x_max = min(int(np.max(x_coords)) + 1, image_width - 1)
    y_max = min(int(np.max(y_coords)) + 1, image_height - 1)

    width = max((x_max - x_min), 1)
    height = max((y_max - y_min), 1)

    # Create a blank binary mask
    binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Fill the polygon on the mask
    cv2.fillPoly(binary_mask, polygon_np, 255)  # 255 for white mask
    # Crop the mask to the bounding box
    binary_mask = binary_mask[y_min:y_max, x_min:x_max]

    bbox = (x_min / image_width, y_min / image_height, width / image_width, height / image_height)
    
    return label, bbox, binary_mask

class Dataset:

    def __init__(self, folder, max_coords=100, mask_size=(40,40)):
        """
        Initialize the dataset.
        """
        self.folder = folder

        # load the classes from classes.txt
        self.classes = []
        with open(os.path.join(self.folder, 'classes.txt'), 'r') as f:
            for line in f:
                self.classes.append(line.strip())
        
        # set up class mappings
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}

        # load the blacklist
        self.id_blacklist = []
        with open(os.path.join(self.folder, 'blacklist.txt'), 'r') as f:
            for line in f:
                self.id_blacklist.append(int(line.strip()))

        # list all files from images folder
        self.train_filepaths = [os.path.join(self.folder, 'images', 'train', f) for f in os.listdir(os.path.join(self.folder, 'images', 'train')) if f.endswith('.png')]
        self.train_gt_files = [f.replace('images', 'labels').replace('.png', '.txt') for f in self.train_filepaths]

        self.val_filepaths = [os.path.join(self.folder, 'images', 'val', f) for f in os.listdir(os.path.join(self.folder, 'images', 'val')) if f.endswith('.png')]
        self.val_gt_files = [f.replace('images', 'labels').replace('.png', '.txt') for f in self.val_filepaths]

        # Filter out blacklisted files
        self.train_filepaths = [f for i, f in enumerate(self.train_filepaths) if i not in self.id_blacklist]
        self.train_gt_files = [f for i, f in enumerate(self.train_gt_files) if i not in self.id_blacklist]
        self.val_filepaths = [f for i, f in enumerate(self.val_filepaths) if i not in self.id_blacklist]
        self.val_gt_files = [f for i, f in enumerate(self.val_gt_files) if i not in self.id_blacklist]

        # create ground truth data structures
        self.train_gt = self._load_gt_files(self.train_gt_files)
        self.val_gt = self._load_gt_files(self.val_gt_files)

        self.max_coords = max_coords
        self.mask_size = mask_size
        self.id_blacklist = set(self.id_blacklist)
        
        # Set up mode (default to train)
        self.filepaths = self.train_filepaths
        self.gt = self.train_gt
        
    def _load_gt_files(self, gt_files):
        """Load ground truth files"""
        result = []
        for gt_file in gt_files:
            try:
                with open(gt_file, 'r') as f:
                    detections = []
                    for line in f:
                        label, bbox, binary_mask = parse_annotation_line(line, 640, 640)
                        detections.append({
                            'label': label,
                            'bounding_box': bbox,
                            'mask': binary_mask
                        })
                    result.append({'detections': detections})
            except:
                continue
        return result
    
    def set_mode(self, mode):
        """Set the dataset mode to train or val"""
        if mode == 'train':
            self.filepaths = self.train_filepaths
            self.gt = self.train_gt
        else:
            self.filepaths = self.val_filepaths
            self.gt = self.val_gt
        
    def __len__(self):
        """Get the dataset length"""
        return len(self.filepaths)
        
    def get_sample(self, idx):
        """Get a sample from the dataset by index"""
        sample_path = self.filepaths[idx]
        
        # Load the image (don't actually load in this API, just return metadata)
        img_filename = os.path.basename(sample_path)
        size = (640, 640)  # Assuming all images are 640x640
        
        # Access detections (bounding boxes, labels)
        try:
            detections = self.gt[idx]['detections']
            bboxes = []
            masks = []
            labels = []
            for det in detections:
                try:
                    bbox = det['bounding_box']  # [x_min, y_min, width, height]
                    label = det['label']
                    other_labels = [ 13, 14, 16, 17  ]
                    probability = [ 0.55, 0.1, 0.2, 0.15 ]
                    if label == 12:
                        if random.random() < 0.2:
                            label = np.random.choice( other_labels, p = probability )

                    mask = det['mask']
                    mask = mask_to_fixed_points( mask, self.max_coords )
                    mask = normalize_mask_points( mask, bbox, size )  # Normalize mask points

                    # Add the model target
                    p0, p1, p2, p3 = bbox
                    bboxes.append( [ p0, p1, p2, p3 ] )
                    labels.append( label )
                    masks.append( mask )
                except:
                    continue
        except:
            # If there are no detections, return empty values
            bboxes = []
            masks = []
            labels = []
    
        return {
            "image_path": img_filename,
            "bboxes": bboxes,
            "labels": labels
        }

def load_dataset():
    """Load the dataset"""
    return Dataset(dataset_folder)

def get_class_statistics(dataset, mode='train'):
    """Get class statistics for the dataset"""
    dataset.set_mode(mode)
    
    # Initialize dictionaries for counts and probabilities
    class_counts = {cls: 0 for cls in dataset.classes}
    class_presence = {cls: 0 for cls in dataset.classes}
    images_with_classes = {cls: [] for cls in dataset.classes}
    
    # Count for each class
    for i in tqdm(range(len(dataset))):
        sample = dataset.get_sample(i)
        seen_classes = set()
        
        for label in sample['labels']:
            class_name = dataset.idx_to_class[label]
            class_counts[class_name] += 1
            seen_classes.add(class_name)
            
        # Track which images have each class
        for cls in seen_classes:
            class_presence[cls] += 1
            images_with_classes[cls].append(i)
    
    # Calculate statistics
    total_images = len(dataset)
    class_stats = []
    
    for cls, count in class_counts.items():
        if cls == 'no_object':
            continue
            
        presence = class_presence[cls]
        percentage = (presence / total_images) * 100 if total_images > 0 else 0
        avg_per_image = count / presence if presence > 0 else 0
        
        class_stats.append({
            'class': cls,
            'total': count,
            'images': presence,
            'percentage': percentage,
            'avg_per_image': avg_per_image
        })
    
    # Sort by total count (descending)
    class_stats.sort(key=lambda x: x['total'], reverse=True)
    
    return {
        'total_images': total_images,
        'class_stats': class_stats
    }

def get_detection_distribution(dataset, mode='train'):
    """Get the distribution of detections per image"""
    dataset.set_mode(mode)
    
    detection_counts = []
    for i in tqdm(range(len(dataset))):
        sample = dataset.get_sample(i)
        detection_counts.append(len(sample['labels']))
    
    # Calculate distribution
    max_detections = max(detection_counts) if detection_counts else 0
    distribution = [0] * (max_detections + 1)
    for count in detection_counts:
        distribution[count] += 1
    
    distribution_data = []
    for i, count in enumerate(distribution):
        if i > 0:  # Skip 0 detections
            distribution_data.append({
                'detections': i,
                'images': count,
                'percentage': (count / len(dataset)) * 100
            })
    
    return {
        'distribution': distribution_data,
        'max_detections': max_detections,
        'avg_detections': sum(detection_counts) / len(dataset) if detection_counts else 0
    }

def get_class_diversity(dataset, mode='train'):
    """Get the diversity of classes per image"""
    dataset.set_mode(mode)
    
    diversity_counts = []
    for i in tqdm(range(len(dataset))):
        sample = dataset.get_sample(i)
        unique_classes = set(dataset.idx_to_class[label] for label in sample['labels'])
        diversity_counts.append(len(unique_classes))
    
    # Calculate distribution
    max_diversity = max(diversity_counts) if diversity_counts else 0
    distribution = [0] * (max_diversity + 1)
    for count in diversity_counts:
        distribution[count] += 1
    
    # Skip the 0 index
    distribution = distribution[1:]
    
    diversity_data = []
    for i, count in enumerate(distribution):
        diversity_data.append({
            'unique_classes': i + 1,  # Add 1 since we skipped index 0
            'images': count,
            'percentage': (count / len(dataset)) * 100
        })
    
    return {
        'distribution': diversity_data,
        'max_diversity': max_diversity,
        'avg_diversity': sum(diversity_counts) / len(dataset) if diversity_counts else 0
    }

def get_spatial_heatmaps(dataset, mode='train'):
    """Generate spatial heatmaps for each class"""
    dataset.set_mode(mode)
    
    # Resolution of each heatmap
    heatmap_resolution = (100, 100)
    image_width, image_height = 640, 640
    
    # Initialize heatmaps for each class
    class_heatmaps = {cls: np.zeros(heatmap_resolution) for cls in dataset.classes if cls != 'no_object'}
    # Also create a combined heatmap of all objects
    all_objects_heatmap = np.zeros(heatmap_resolution)
    
    # Count objects in each location
    for i in tqdm(range(len(dataset))):
        sample = dataset.get_sample(i)
        for bbox, label in zip(sample['bboxes'], sample['labels']):
            class_name = dataset.idx_to_class[label]
            if class_name == 'no_object':
                continue
            
            # Calculate object center in absolute coordinates
            x_center = bbox[0] + bbox[2]/2
            y_center = bbox[1] + bbox[3]/2
            
            # Convert to heatmap grid coordinates
            grid_x = int(x_center * heatmap_resolution[1])
            grid_y = int(y_center * heatmap_resolution[0])
            
            # Ensure coordinates are within bounds
            grid_x = min(max(grid_x, 0), heatmap_resolution[1]-1)
            grid_y = min(max(grid_y, 0), heatmap_resolution[0]-1)
            
            # Increment count at this location
            class_heatmaps[class_name][grid_y, grid_x] += 1
            all_objects_heatmap[grid_y, grid_x] += 1
    
    # Convert to base64 encoded PNG images
    import io
    import base64
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    
    heatmap_images = {}
    
    # First process the combined heatmap
    if np.max(all_objects_heatmap) > 0:
        normalized = all_objects_heatmap / np.max(all_objects_heatmap)
        
        # Create matplotlib figure
        fig = plt.figure(figsize=(4, 4), dpi=100)
        plt.imshow(normalized, cmap='hot', interpolation='nearest')
        plt.title('All Objects Combined')
        plt.colorbar(label='Normalized Density')
        plt.axis('off')
        
        # Save to in-memory file
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        heatmap_images['__all__'] = img_str
    
    # Process individual class heatmaps
    for class_name, heatmap in class_heatmaps.items():
        if np.max(heatmap) > 0:  # Only process non-empty heatmaps
            # Normalize for visualization
            normalized = heatmap / np.max(heatmap)
            
            # Create matplotlib figure
            fig = plt.figure(figsize=(3, 3), dpi=100)
            plt.imshow(normalized, cmap='hot', interpolation='nearest')
            plt.title(class_name)
            plt.axis('off')
            
            # Save to in-memory file
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            heatmap_images[class_name] = img_str
    
    return heatmap_images

def get_dataset_summary(mode='train'):
    """Get a summary of the dataset"""
    dataset = load_dataset()
    
    class_stats = get_class_statistics(dataset, mode)
    detection_dist = get_detection_distribution(dataset, mode)
    class_diversity = get_class_diversity(dataset, mode)
    spatial_heatmaps = get_spatial_heatmaps(dataset, mode)
    
    return {
        'mode': mode,
        'total_images': class_stats['total_images'],
        'class_stats': class_stats['class_stats'],
        'detection_distribution': detection_dist['distribution'],
        'max_detections': detection_dist['max_detections'],
        'avg_detections': detection_dist['avg_detections'],
        'class_diversity': class_diversity['distribution'],
        'max_diversity': class_diversity['max_diversity'],
        'avg_diversity': class_diversity['avg_diversity'],
        'classes': dataset.classes,
        'spatial_heatmaps': spatial_heatmaps
    }

def api_dataset_summary(mode='train'):
    """API endpoint to get dataset summary"""
    summary = get_dataset_summary(mode)
    return jsonify(summary)