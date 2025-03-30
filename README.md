# Two-Stage Encoder-Detection Model

## Project Overview

This project implements a novel two-stage vision model architecture that combines:

1. **Flexible Image Encoder**: A resolution-agnostic image encoder that extracts rich visual features
2. **Anchor-Free Detection**: A detection system inspired by DETR (DEtection TRansformer) models

## Key Features

- **Resolution Flexibility**: Works with any image resolution without requiring resizing
- **Anchor-Free Design**: Eliminates the need for predefined anchor boxes
- **Multi-Scale Feature Processing**: Processes features at multiple scales for improved detection
- **Self-Supervised Learning**: Utilizes techniques like DINO and iBOT for self-supervised pretraining
- **Transformer Architecture**: Leverages transformer-based design for improved context understanding

## Architecture

The model consists of two main components:

### 1. Backbone Encoder
- Convolutional backbone
- Vision transformer layers
- Multi-scale feature extraction
- Self-supervised pretraining with DINO, iBOT, and VicReg losses

### 2. Detection Module
- Transformer detection head
- Proposal generation
- Hungarian matcher for loss computation
- Multi-scale feature refinement

## Applications

This model is designed for:
- Object detection in complex scenes
- Working with varying image resolutions
- Transfer learning to downstream tasks
- Applications where traditional anchor-based detectors struggle with unusual object shapes

## Implementation Details

The implementation combines several state-of-the-art techniques:
- DINOv2 for teacher-student learning
- Transformer architecture for feature processing
- Hungarian matching for object assignment
- Focal loss and IoU-based loss functions

## Getting Started

```bash
# Install requirements
pip install -r requirements.txt

# Train encoder model
python train.py

# Train detection model
python train_detection.py
```

## Project Structure
- `models.py`: Model architecture definitions
- `layers.py`: Building blocks and layer implementations
- `losses.py`: Loss functions
- `train.py`: Training script for encoder
- `train_detection.py`: Training script for detector
- `data.py`: Data loading utilities
- `utils.py`: Helper functions and utilities