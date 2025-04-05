import random
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
from PIL import ImageDraw
import time
import torch
import numpy as np

def sample_and_draw(dataset, num_samples=16, grid_rows=4, grid_cols=4):
    """
    Randomly samples num_samples items from the dataset, draws bounding boxes and polygons,
    and returns (and optionally saves) a grid image of the results.
    
    Args:
        dataset (Dataset): Your DeficiencyDataset instance.
        num_samples (int): Total number of samples to draw (should equal grid_rows * grid_cols).
        grid_rows (int): Number of rows in the grid.
        grid_cols (int): Number of columns in the grid.
        output_path (str, optional): If provided, the grid image will be saved to this path.
        
    Returns:
        grid_img (PIL.Image): The resulting grid image with drawn annotations.
    """
    # We'll use ToPILImage to convert tensor images (if needed)
    to_pil = ToPILImage()
    to_tensor = ToTensor()

    drawn_images = []
    # Randomly select indices from the dataset.
    indices = random.sample( range( len( dataset ) ), num_samples )
    start = time.time()
    data = [ dataset[i] for i in indices ]
    print( f"Time taken to load {num_samples} samples: {time.time() - start:.2f} seconds" )
    poligons_shapes =[]
    for img, annotations in data:
        
        # If the image is a tensor, convert it to a PIL image.
        if isinstance(img, torch.Tensor):
            # Assume image tensor is in [C,H,W] format.
            img = to_pil( img )
        
        # Create a drawing context.
        draw = ImageDraw.Draw( img )

        # The crop size is assumed to be the image size.
        crop_w, crop_h = img.size
        for ann in annotations:
            # Unpack annotation: bbox is normalized, polygon is given as np.array.
            label = ann["label"]
            bbox = ann["bbox"]  # (x_min, y_min, w, h), normalized relative to crop
            polygon_np = ann["polygon"]  # shape [1, num_points, 2], integer coordinates relative to crop
            poligons_shapes.append( polygon_np.shape )
            
            # Convert normalized bbox to absolute coordinates.
            x_min = bbox[0] * crop_w
            y_min = bbox[1] * crop_h
            x_max = ( bbox[0] + bbox[2] ) * crop_w
            y_max = ( bbox[1] + bbox[3] ) * crop_h
            
            # Draw bounding box.
            draw.rectangle( [ x_min, y_min, x_max, y_max ], outline = "red", width = 2 )
                        
            # Draw polygon if available.
            if polygon_np is not None and polygon_np.shape[1] >= 3:
                # polygon_np is expected in OpenCV format: shape [1, num_points, 2].
                poly_points = [tuple(pt) for pt in polygon_np[0]]
                draw.line( poly_points + [ poly_points[0] ], fill = "blue", width = 2 )

        # Append the drawn image.
        drawn_images.append(to_tensor(img))
    
    # Create a grid of images.
    grid = make_grid(drawn_images, nrow=grid_cols, padding=4)
    grid_img = to_pil(grid)
    
    return grid_img, np.unique( poligons_shapes )

