from src.data import build_data_loader_train_detection, denormalize_transform

import os
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics import RTDETR
import cv2
import torchvision.transforms as T
import numpy as np
import gc

data_loader = build_data_loader_train_detection( 'dataset_detection', batch_size = 8, max_objects = 200, max_poly_points = 64, crop_size = 640, mode = 'seg', train = True )

MODE = 'val'
BASE_FOLDER = "dataset_detection_pre"
OUTPUT_IMAGE_FOLDER = f'images/{MODE}'
OUTPUT_ANNOTATED_IMAGE_FOLDER = f'annotated_images/{MODE}'
OUTPUT_LABEL_FOLDER = f'labels/{MODE}'
TRAIN_TXT_PATH = f'{MODE}.txt'
DATA_YAML_PATH = 'data.yaml'

os.makedirs( os.path.join( BASE_FOLDER, OUTPUT_IMAGE_FOLDER ), exist_ok = True )
os.makedirs( os.path.join( BASE_FOLDER, OUTPUT_ANNOTATED_IMAGE_FOLDER ), exist_ok = True )
os.makedirs( os.path.join( BASE_FOLDER, OUTPUT_LABEL_FOLDER ), exist_ok = True )

y12 = YOLO( 'yolo/yolo12x.pt', task = 'detect' )
y11 = YOLO( 'yolo/yolo11x.pt', task = 'detect' )
dterr = RTDETR( "yolo/rtdetr-x.pt" )
to_pil = T.ToPILImage()
train_txt_lines = []

yolo_to_custom = {
    0: 1,    # person ➜ Person
    1: 9,    # bicycle ➜ Bicycle
    2: 10,   # car ➜ LMVs
    3: 8,    # motorcycle ➜ Motorcycle
    5: 11,   # bus ➜ HMVs
    6: 11,   # train ➜ HMVs
    7: 11,   # truck ➜ HMVs
    9: 7,    # traffic light ➜ Traffic Light
    10: 6,   # fire hydrant ➜ Fire Hydrant
    11: 4,   # stop sign ➜ Stop Sign
    12: 3,   # parking meter ➜ Parking Meter
    14: 2,   # bird ➜ Birds
    16: 12,  # dog ➜ Animals
    17: 12,  # horse ➜ Animals
    18: 12,  # sheep ➜ Animals
    19: 12,  # cow ➜ Animals
    20: 12,  # elephant ➜ Animals
    21: 12,  # bear ➜ Animals
    22: 12,  # zebra ➜ Animals
    23: 12,  # giraffe ➜ Animals
    56: 20,  # chair ➜ Furniture
    58: 21,  # potted plant ➜ Pot Plant
    13: 20,  # bench ➜ Furniture
    33: 22,  # kite ➜ Sign Boards (loosely)
    45: 20,  # bowl ➜ Furniture
}

def compute_iou_center(box1, box2):
    """
    Compute IoU for two boxes in [x_left, y_top, w, h] (normalized) format.
    """

    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2

    box1_x1, box1_y1 = ( cx1 - w1 / 2 ), ( cy1 - h1 / 2 )
    box1_x2, box1_y2 = ( cx1 + w1 / 2 ), ( cy1 + h1 / 2 )

    box2_x1, box2_y1 = cx2 - w2 / 2, cy2 - h2 / 2
    box2_x2, box2_y2 = cx2 + w2 / 2, cy2 + h2 / 2

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    if union == 0:
        return 0
    return inter_area / union

def area_from_center(box_center):
    """
    Calculate area of a box given in [cx, cy, w, h] format.
    """
    cx, cy, w, h = box_center
    return w * h

def deduplicate_detections(detections, iou_threshold=0.5):
    """
    Remove duplicate detections from a list by keeping the larger detection.
    
    detections: list of tuples (custom_class, box)
      where box is in [x_left, y_top, w, h] (normalized) format.
    iou_threshold: if IoU between two boxes of the same class exceeds this threshold,
                   they are considered duplicates.
                   
    Returns:
      A deduplicated list of detections, keeping the detection with the larger area
      for overlapping detections.
    """
    deduped = []
    for det in detections:
        cls_det, box_det, _ = det
        area_det = area_from_center( box_det )  # area = w * h
        duplicate_found = False
        for idx, d in enumerate(deduped):
            cls_existing, box_existing, _ = d
            if cls_existing == cls_det:
                iou = compute_iou_center(box_det, box_existing)
                if iou > iou_threshold:
                    area_existing = area_from_center( box_existing )
                    # If new detection has a bigger area, replace the existing one.
                    if area_det > area_existing:
                        deduped[idx] = det
                    duplicate_found = True
                    break
        if not duplicate_found:
            deduped.append(det)
    return deduped

def apply_nsm(detections, iou_threshold=0.5, confidence_threshold=0.25):
    """
    Apply Non-Maximum Suppression to detections.
    
    Parameters:
    - detections: list of tuples (class, box, [optional]confidence), where box is in [x_left, y_top, w, h] format
    - iou_threshold: threshold for IoU to consider boxes as overlapping
    - confidence_threshold: confidence threshold for filtering detections
    
    Returns:
    - List of detections after NMS
    """
    if not detections:
        return []
    
    # Check if detections have confidence values attached
    has_confidence = False
    if len(detections) > 0 and isinstance(detections[0], tuple):
        if len(detections[0]) > 2:  # (class, box, confidence)
            has_confidence = True
        
    # Filter by confidence if available
    filtered_detections = []
    for det in detections:
        if has_confidence:
            cls, box, conf = det
            if conf >= confidence_threshold:
                filtered_detections.append((cls, box, conf))
        else:
            filtered_detections.append(det)
            
    if not filtered_detections:
        return []
        
    # Group detections by class
    class_detections = {}
    for det in filtered_detections:
        if has_confidence:
            cls, box, conf = det
            if cls not in class_detections:
                class_detections[cls] = []
            # Use both area and confidence for sorting
            area = area_from_center( box )
            score = area * conf  # score combines size and confidence
            class_detections[cls].append((box, score, conf))
        else:
            cls, box = det
            if cls not in class_detections:
                class_detections[cls] = []
            # Calculate area for sorting
            area = area_from_center( box )
            class_detections[cls].append((box, area, 1.0))  # Default confidence of 1.0
    
    # Apply NMS for each class separately
    kept_detections = []
    for cls, dets in class_detections.items():
        # Sort by score/area (descending)
        sorted_dets = sorted(dets, key=lambda x: x[1], reverse=True)
        
        nms_boxes = []
        for box, _, conf in sorted_dets:
            # Check if this box overlaps with any selected box
            keep = True
            for kept_box in nms_boxes:
                iou = compute_iou_center(box, kept_box)
                if iou > iou_threshold:
                    keep = False
                    break
            
            if keep:
                nms_boxes.append(box)
                if has_confidence:
                    kept_detections.append((cls, box, conf))
                else:
                    kept_detections.append((cls, box))
    
    return kept_detections

def merge_detections(result, ds_labels, ds_boxes, yolo_to_custom, iou_threshold=0.5):
    """
    Merges dataset detections with YOLO detections for one image.

    - result: YOLO detection result for one image (each box has .cls and .xywhn)
    - ds_labels: tensor of shape [N] containing dataset detection labels (0 means no detection)
    - ds_boxes: tensor of shape [N, 4] containing dataset boxes in [x_left, y_top, w, h] normalized format
    - yolo_to_custom: dictionary mapping YOLO class ids to your custom class ids
    - iou_threshold: IoU threshold for matching

    For detections with the same label and IoU > iou_threshold, only the detection with the larger area is kept.

    Returns:
      A list of tuples (custom_class, box) where box is in [x_left, y_top, w, h] normalized format.
      Dataset detections are kept, and a YOLO detection is added only if it does not conflict
      with an existing dataset detection of the same class. In case of conflict, the bigger detection is kept.
    """
    final_detections = []
    
    # 1. Add dataset detections (non-zero labels) as they are.
    for label, box in zip(ds_labels, ds_boxes):
        if label.item() != 0:
            final_detections.append( ( int( label.item() ) - 1, box.tolist(), 1.0 ) )
    
    # 2. Process YOLO detections.
    for yolo_box in result.boxes:
        yolo_cls = int( yolo_box.cls )
        if yolo_cls not in yolo_to_custom:
            continue
        custom_cls = yolo_to_custom[yolo_cls] - 1
        # Get YOLO detection in center format and convert to top-left format.
        yolo_center = yolo_box.xywhn[0].tolist()  # [cx, cy, w, h]
        area_new = area_from_center( yolo_center )  # area of new detection
        confidence = yolo_box.conf[0].item()
        if confidence < 0.5:
            continue
        
        conflict_found = False
        indices_to_remove = []
        # Check for conflict with any existing detection of the same class.
        for idx, (existing_cls, existing_box, _) in enumerate(final_detections):            
            iou = compute_iou_center( yolo_center, existing_box )
            if iou > iou_threshold and confidence < 0.7:
                conflict_found = True
                break
            if existing_cls == custom_cls:
                if iou > iou_threshold:
                    area_existing = area_from_center( existing_box )
                    if area_new > area_existing:
                        # New detection is bigger, mark the existing one for removal.
                        indices_to_remove.append(idx)
                    else:
                        # Existing detection is bigger; skip adding the new one.
                        conflict_found = True
                        break
        if not conflict_found:
            # Remove any conflicting detections that are smaller.
            for idx in sorted(indices_to_remove, reverse=True):
                del final_detections[idx]
            final_detections.append((custom_cls, yolo_center, confidence))  # Include confidence
    
    return final_detections

i = 0
max_images = 2000
for batch in data_loader:
    # Convert tensor images to NumPy arrays (and multiply by 255 if your transform normalizes to [0,1])
    images = list(batch[0].unbind(0))
    images = [ img.permute(1, 2, 0).cpu().numpy() * 255 for img in images ]
    
    # Get dataset detections:
    # ds_labels: tensor of shape [batch_size, N] with labels (in [x_left, y_top, w, h] format for boxes)
    # ds_boxes: tensor of shape [batch_size, N, 4] with boxes in [x_left, y_top, w, h] normalized format
    ds_labels = list( batch[1].unbind(0) )
    ds_boxes = list( batch[2].unbind(0) )
    
    r12 = y12.predict( images, verbose = False, conf = 0.25, iou = 0.45, agnostic_nms = True )
    r11 = y11.predict( images, verbose = False, conf = 0.25, iou = 0.45, agnostic_nms = True )
    rtdetr = dterr.predict( images, verbose = False )

    for idx in range(len(images)):

        final_detections_1 = merge_detections( rtdetr[idx], ds_labels[idx], ds_boxes[idx], yolo_to_custom, iou_threshold = 0.3 )
        final_detections_2 = merge_detections( r12[idx], ds_labels[idx], ds_boxes[idx], yolo_to_custom, iou_threshold = 0.3 )
        final_detections_3 = merge_detections( r11[idx], ds_labels[idx], ds_boxes[idx], yolo_to_custom, iou_threshold = 0.3 )

        # Combine detections from all three sources
        combined_detections = final_detections_1 + final_detections_2 + final_detections_3

        # First deduplicate overlapping detections of the same class
        final_detections = deduplicate_detections( combined_detections, iou_threshold = 0.1 )
        
        # Then apply Non-Maximum Suppression (NSM) to further refine detections
        final_detections = apply_nsm( final_detections, iou_threshold = 0.3, confidence_threshold = 0.5 )

        # Skip sample if no final detections.
        if not final_detections:
            continue

        filename = f'image_{i}.png'
        output_annotated_img_path = os.path.join(BASE_FOLDER, OUTPUT_ANNOTATED_IMAGE_FOLDER, filename)
        output_img_path = os.path.join(BASE_FOLDER, OUTPUT_IMAGE_FOLDER, filename)
        
        image = images[idx]
        annotated_image = image.copy()
        h_img, w_img, _ = annotated_image.shape
        
        # Draw each final detection on the image.
        for custom_cls, box, _ in final_detections:
            # box is in [x_left, y_top, w, h] (normalized).
            # Convert normalized coordinates to absolute pixel values.

            x_center, y_center, width, height = box
            x_min = int( ( x_center - width / 2 ) * w_img )
            y_min = int( ( y_center - height / 2 ) * h_img )
            x_max = int( ( x_center + width / 2 ) * w_img )
            y_max = int( ( y_center + height / 2 ) * h_img )

            if custom_cls > 23:
                a = 10
            
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(annotated_image, str(custom_cls), (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save annotated image.
        cv2.imwrite(output_annotated_img_path, annotated_image)
        cv2.imwrite(output_img_path, cv2.cvtColor( image, cv2.COLOR_BGR2RGB ) )
        
        # Save label file in YOLO format but with your box format ([x_left, y_top, w, h]).
        label_path = os.path.join(BASE_FOLDER, OUTPUT_LABEL_FOLDER, Path(filename).stem + '.txt')
        with open(label_path, 'w') as f:
            for custom_cls, box, _ in final_detections:
                f.write(f"{custom_cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
        
        train_txt_lines.append(str(Path(output_annotated_img_path).resolve()))
        with open( os.path.join( BASE_FOLDER, TRAIN_TXT_PATH ), 'a' ) as f:
            f.write( str(Path(output_annotated_img_path).resolve()) + '\n' )
        print(f"{i} of {max_images}", end='\r')

        del annotated_image        
        del final_detections
        gc.collect()
        torch.cuda.empty_cache()

        i += 1
        if i >= max_images:
            break
    
    del rtdetr
    del r12
    del r11
    del images
    del ds_labels
    del ds_boxes
    gc.collect()
    torch.cuda.empty_cache()
    if i >= max_images:
        break


