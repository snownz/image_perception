from ultralytics import RTDETR
import torch

model = RTDETR('yolo/rtdetr-x.pt')

# model( 'dataset_detection/images/train/front_1706545937.png' )

model.train(
    data = 'dataset_detection/data.yaml',  # Path to your dataset configuration file
    # data = 'dataset_detection/data.yaml',  # Path to your dataset configuration file
    epochs = 20,
    imgsz = 640,
    batch = 16,
    project = 'yolo/runs/train',
    name = 'rtdetr_custom_test',
)