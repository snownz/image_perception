import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F
import os
import numpy as np
import random
from typing import Any
import math
from functools import partial
import itertools
import torchvision.io as io
import numpy as np
from shapely.geometry import Polygon, box, LineString
import os
import random
import torch
import cv2
from torchvision.ops import nms

class Decoder:
    def decode(self):
        raise NotImplementedError

def decode(rgb_path, depth_path):
    # Read the RGB image and convert it to float normalized tensor
    rgb_tensor = io.read_image( rgb_path ).float() / 255.0
    return rgb_tensor
    # if os.path.exists(depth_path):
    #     depth_tensor = io.read_image( depth_path, mode = io.ImageReadMode.GRAY ).float()
    # else:
    #     depth_tensor = torch.zeros( 1, rgb_tensor.shape[1], rgb_tensor.shape[2] )
    # combined = torch.cat( ( rgb_tensor, depth_tensor ), dim = 0 )
    # return combined

class TargetDecoder(Decoder):

    def __init__(self, target):
        self._target = target

    def decode(self) -> Any:
        """
        Decode the target data.
        """
        # Placeholder implementation
        return self._target

class ExtendedVisionDataset(VisionDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore

    def get_image_data(self, index):
        raise NotImplementedError

    def get_target(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        try:
            image_data = self.get_image_data( index )
            image = ImageDataDecoder( image_data ).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target( index )
        target = TargetDecoder( target ).decode()

        if self.transforms is not None:
            image, target = self.transforms( image, target )

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError

class KeverasDataset(ExtendedVisionDataset):

    def __init__(self, root, transforms=None):

        super().__init__( root = root, transforms = transforms )
        self._entries = []
        self._load_entries()

        print(f"Dataset loaded with {len(self)} entries.")

    def _load_entries(self):

        """Parse the dataset directory structure to populate the entries."""
        for street_name in os.listdir( self.root ):
            street_path = os.path.join( self.root, street_name )
            if not os.path.isdir( street_path ):
                continue

            for number in os.listdir( street_path ):
                number_path = os.path.join( street_path, number )
                if not os.path.isdir( number_path ):
                    continue

                color_dir = os.path.join( number_path, "color" )
                depth_dir = os.path.join( number_path, "depth" )

                if not os.path.exists( color_dir ) or not os.path.exists( depth_dir ):
                    continue

                # for position in [ "left", "front", "rear_left", "right", "rear_right" ]:
                for position in [ "left", "front", "right" ]:

                    position_color_dir = os.path.join( color_dir, position )

                    if not os.path.exists( position_color_dir ):
                        continue

                    color_image_paths = [ os.path.join( position_color_dir, image ) for image in os.listdir( position_color_dir ) if image.endswith( ".png" ) ]

                    for color_image_path in color_image_paths:

                        self._entries.append( {
                            "color_path": color_image_path,
                            "depth_path": color_image_path.replace( "color", "depth" ),
                        } )

    def get_image_data(self, index: int) -> tuple:
        """
        Return raw binary data for the RGB and depth images.
        """
        entry = self._entries[index]
        return entry["color_path"], entry["depth_path"]

    def get_target(self, index: int):
        """Return None as this dataset has no labels."""
        return None

    def __len__(self) -> int:
        """Return the total number of entries."""
        return len( self._entries )

    def __getitem__(self, index: int):

        """
        Use the ImageDataDecoder to decode the image data and apply transformations.
        """
        image_data = self.get_image_data( index )
        image = decode( *image_data )
        target = self.get_target( index )  # Always None for this dataset
        if self.transforms is not None:
            image = self.transforms( image ) # too slow to augment

        return image, target

class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)

class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)

class DynamicThresholdSolarize:

    def __init__(self, threshold=128, p=0.5):
        self.threshold = threshold
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            img_array = np.array( img )
            img_max = max( img_array.max().item(), 0 ) # Get the maximum value of the image
            effective_threshold = min(self.threshold, img_max)  # Adjust threshold dynamically
            try: return F.solarize(img, effective_threshold)
            except: return img
        return img

# IMAGENET_DEFAULT_MEAN = ( 0.5, 0.5, 0.5 )
# IMAGENET_DEFAULT_STD = ( 0.5, 0.5, 0.5 )

IMAGENET_DEFAULT_MEAN = [ 0.485, 0.456, 0.406 ]
IMAGENET_DEFAULT_STD = [ 0.229, 0.224, 0.225 ]

def make_normalize_transform(
    mean= IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize( mean = mean, std = std )

def make_denormalize_transform(
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)

# denormalize_transform = lambda x: x * 0.5 + 0.5 
denormalize_transform = lambda x: make_denormalize_transform()( x )

class DataAugmentation(object):

    def __init__(
        self,
        global_crops_scale,
        global_crops_size=224,
    ):
        self.global_crops_scale = global_crops_scale
        self.global_crops_size = global_crops_size

        print("###################################")
        print("Using data augmentation parameters:")
        print(f"global_crops_scale: {global_crops_scale}")
        print(f"global_crops_size: {global_crops_size}")
        print("###################################")

        self.geometric_augmentation_global = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size, scale = global_crops_scale, interpolation = transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip( p = 0.5 ),
        ])

        # color distortions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness = 0.4, contrast = 0.4, saturation = 0.2, hue = 0.1
                        )
                    ],
                    p = 0.4,
                ),
                transforms.RandomGrayscale( p = 0.2 ),
            ]
        )

        global_transfo1_extra = GaussianBlur( p = 0.5 )

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur( p = 0.1 )
            ]
        )

        # self.global_transfo1 = transforms.Compose( [ color_jittering, global_transfo1_extra, make_normalize_transform() ] )
        # self.global_transfo2 = transforms.Compose( [ color_jittering, global_transfo2_extra, make_normalize_transform() ] )

        self.global_transfo1 = transforms.Compose( [ color_jittering, global_transfo1_extra ] )
        self.global_transfo2 = transforms.Compose( [ color_jittering, global_transfo2_extra ] )

        # self.global_transfo1 = transforms.Compose( [ make_normalize_transform() ] )
        # self.global_transfo2 = transforms.Compose( [ make_normalize_transform() ] )

    def _apply_color_transform(self, image, transform):

        """Apply color transformations only to the RGB channels."""
        return transform( image )

    def _apply_geometric_transform(self, image, transform):
        """Apply geometric transformations to the image."""
        return transform( image )
    
    def __call__(self, image):

        output = {}

        image_array = image

        # global crops:
        im1_base = self._apply_geometric_transform( image_array, self.geometric_augmentation_global )  # Global crop 1
        im2_base = self._apply_geometric_transform( image_array, self.geometric_augmentation_global )  # Global crop 2

        global_crop_1 = self._apply_color_transform( im1_base, self.global_transfo1 )  # Color transformation
        global_crop_2 = self._apply_color_transform( im2_base, self.global_transfo2 )  # Color transformation

        while torch.isnan(global_crop_1).any():
            global_crop_1 = self._apply_color_transform( im1_base, self.global_transfo1 )
        
        if torch.isnan(global_crop_2).any():
            global_crop_2 = self._apply_color_transform( im2_base, self.global_transfo2 )

        output["global_crops"] = [ global_crop_1, global_crop_2 ]

        return output

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):

    collated_global_crops_1 = []
    collated_global_crops_2 = []
    for s in samples_list:
        crop = s[0]["global_crops"]
        collated_global_crops_1.append( crop[0] )
        collated_global_crops_2.append( crop[1] )
    collated_global_crops_1 = torch.stack( collated_global_crops_1 ) # [ B, C, H, W ]
    collated_global_crops_2 = torch.stack( collated_global_crops_2 ) # [ B, C, H, W ]
    collated_global_crops = torch.cat( [ collated_global_crops_1, collated_global_crops_2 ] ) # [ 2*B, C, H, W ]

    B = len( collated_global_crops ) # Batch size
    N = n_tokens # [ 64*64, 32*32, 16*16 ]
    n_samples_masked = int( B * mask_probability ) # Number of samples to be masked
    probs = torch.linspace( *mask_ratio_tuple, n_samples_masked + 1 ) # Linearly spaced probabilities
    upperbound = [ 0 ] * len( N )
    masks_list = [ ]
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        for j, nv in enumerate( N ):
            if len( masks_list ) <= j:
                masks_list.append( [ ] )
            masks_list[j].append( torch.BoolTensor( mask_generator[j]( int( nv * random.uniform( prob_min, prob_max ) ) ) ) )
            upperbound[j] += int( nv * prob_max )
    for i in range( n_samples_masked, B ):
        for j, nv in enumerate( N ):
            masks_list[j].append( torch.BoolTensor( mask_generator[j](0) ) ) # No masking

    final_masks = []
    mask_indices_list = []
    n_masked_patches = []
    masks_weight = []
    for masks in masks_list:
        random.shuffle( masks )
        masks = torch.stack( masks ).flatten(1) # [ B, N ]
        final_masks.append( masks )
        indices = masks.flatten().nonzero().flatten() # [ B*N ]
        mask_indices_list.append( indices )
        n_masked_patches.append( torch.full( (1,), fill_value = indices.shape[0], dtype = torch.long ) )
        masks_weight.append( ( 1 / masks.sum( -1 ).clamp( min = 1.0 ) ).unsqueeze(-1).expand_as( masks )[masks] )

    del masks_list

    return {
        "collated_global_crops": collated_global_crops.to( dtype ),
        "collated_masks": final_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": n_masked_patches,
    }

class MaskingGenerator:

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance( input_size, tuple ):
            input_size = ( input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = ( math.log( min_aspect ), math.log( max_aspect ) )

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):

            target_area = random.uniform( self.min_num_patches, max_mask_patches )
            aspect_ratio = math.exp( random.uniform( *self.log_aspect_ratio ) )
            h = int( round( math.sqrt( target_area * aspect_ratio ) ) )
            w = int( round( math.sqrt( target_area / aspect_ratio ) ) )

            if w < self.width and h < self.height:
                top = random.randint( 0, self.height - h )
                left = random.randint( 0, self.width - w )

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):

        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0

        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

def _get_torch_dtype(size: int) -> Any:
    return torch.int32 if size <= 2**31 else torch.int64

def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = _get_torch_dtype(size)
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()
        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value

class InfiniteSampler(Sampler):

    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._advance = advance
        self._start = 0
        self._step = 1

    def __iter__(self):
        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice( iterator, self._advance, None )

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range( self._sample_count )
            yield from itertools.islice( iterable, self._start, None, self._step )

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator().manual_seed( self._seed )

        while True:
            iterable = _generate_randperm_indices( size = self._sample_count, generator = generator )
            yield from itertools.islice( iterable, self._start, None, self._step )

def build_data_loader(root, batch_size, global_crops_size=512, random=True):

    transforms = DataAugmentation( global_crops_scale = ( 0.32, 1.0 ), global_crops_size = global_crops_size )

    dataset =  KeverasDataset( root = root, transforms = transforms )

    mask_generators = [
        MaskingGenerator( 64, num_masking_patches = 64*64 ),
        MaskingGenerator( 32, num_masking_patches = 32*32 ),
        MaskingGenerator( 16, num_masking_patches = 16*16 ),
        # MaskingGenerator( 8, num_masking_patches = 8*8 ),
    ]

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple = ( 0.15, 0.5 ),
        mask_probability = 0.5,
        dtype = torch.float32,
        n_tokens = [ 64*64, 32*32, 16*16 ],
        # n_tokens = [ 32*32, 16*16, 8*8 ],
        mask_generator = mask_generators
    )

    if random:
        sampler = InfiniteSampler( sample_count = len( dataset ), shuffle = True, seed = 37, advance = 1 )
        loader =  DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = sampler,
            collate_fn = collate_fn,
            # num_workers = 8, # High number of workers since CPU is not maxed
            # persistent_workers = True,  # Keep workers alive
            # pin_memory = True, # Faster CPU → GPU transfers
            # prefetch_factor = 4, # Load more batches in advance
            # timeout = 60,  # Prevent workers from resetting if disk is slow
            # drop_last = True  # Avoid uneven batch issues
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = False,
            collate_fn = collate_fn,
            num_workers = 8, # High number of workers since CPU is not maxed
            persistent_workers = True,  # Keep workers alive
            pin_memory = True, # Faster CPU → GPU transfers
            prefetch_factor = 2, # Load more batches in advance
            timeout = 60,  # Prevent workers from resetting if disk is slow
            drop_last = True  # Avoid uneven batch issues
        )

    return loader

# --------------------------------------------------------------
# Detection Part
# --------------------------------------------------------------

def convert_to_cxcywh(box):
    # box is xleft, ytop, width, height
    xleft, ytop, width, height = box
    x_center = xleft + width / 2
    y_center = ytop + height / 2
    return ( x_center, y_center, width, height )

def resample_polygon(poly, num_points):
    """
    Resample a polygon so that it has exactly num_points unique vertices.
    It interpolates along the polygon's exterior (which is assumed closed).
    """
    coords = list(poly.exterior.coords)
    # Remove duplicate closing point if present.
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    
    # Create a LineString from the polygon's exterior.
    line = LineString(coords + [coords[0]])
    total_length = line.length
    # Compute equally spaced distances along the perimeter.
    distances = np.linspace(0, total_length, num_points + 1)[:-1]  # omit duplicate closing point
    new_coords = [line.interpolate(d).coords[0] for d in distances]
    # Close the polygon.
    new_coords.append(new_coords[0])
    return Polygon(new_coords)

def simplify_polygon_to_max_points(poly, max_points, iterations=10):
    """
    Adjust the polygon so that it has exactly max_points vertices (unique, excluding closure).
    Uses simplification if the polygon has too many points, or densification (via resampling)
    if it has too few.
    """
    # Remove duplicate closing point if present.
    coords = list(poly.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    current_points = len(coords)
    
    # If already the correct number of points, just return a properly closed polygon.
    if current_points == max_points:
        return Polygon(coords + [coords[0]])
    
    # If too many points, simplify using binary search on the tolerance parameter.
    if current_points > max_points:
        minx, miny, maxx, maxy = poly.bounds
        diag = ((maxx - minx)**2 + (maxy - miny)**2)**0.5
        low = 0.0
        high = diag
        best_poly = poly  # fallback
        for _ in range(iterations):
            mid = (low + high) / 2.0
            poly_simp = poly.simplify(mid, preserve_topology=True)
            simp_coords = list(poly_simp.exterior.coords)
            if len(simp_coords) > 1 and simp_coords[0] == simp_coords[-1]:
                simp_coords = simp_coords[:-1]
            num_points = len(simp_coords)
            if num_points > max_points:
                low = mid  # increase tolerance to simplify more
            else:
                best_poly = poly_simp
                high = mid  # decrease tolerance to retain more detail
        poly = best_poly  # use the best simplified version
    
    # Whether we simplified (or the original polygon had too few points),
    # now resample the polygon so it has exactly max_points vertices.
    poly_resampled = resample_polygon(poly, max_points)
    
    return poly_resampled

def parse_seg_annotation_line_for_crop(line, image_width, image_height, crop_box, area_thresh=0.5, max_points=None, mask_shapes=[]):
    """
    Parses an annotation line in YOLO segmentation format for a full image,
    adjusts the annotation for a given crop, and optionally simplifies the polygon
    to have at most max_points vertices.
    
    Args:
        line (str): The annotation line as a string (e.g. "label x1 y1 x2 y2 ... xn yn").
        image_width (int): Original image width in pixels.
        image_height (int): Original image height in pixels.
        crop_box (tuple): A tuple (x_offset, y_offset, crop_width, crop_height) defining the crop.
        area_thresh (float): Minimum ratio of the intersection area to the original polygon area for the detection to be kept.
        max_points (int, optional): Maximum number of vertices allowed in the output polygon. If None, no simplification is done.
        
    Returns:
        (int, tuple, numpy.ndarray) or None:
            - label (int): The class label.
            - new_bbox (tuple): Bounding box for the crop in normalized coordinates 
                                (x_min, y_min, width, height), relative to the crop.
            - new_polygon_np (numpy.ndarray): Array of new polygon points in integer pixel values (OpenCV format).
            Returns None if the polygon does not meet the area threshold or is invalid.
    """
    # Split the annotation line.
    parts = line.strip().split(' ')
    if len(parts) < 1:
        return False    
    
    # Extract label.
    label = int(parts[0])
    
    # Extract normalized polygon coordinates.
    mask_coords_norm = list(map(float, parts[1:]))
    if len(mask_coords_norm) % 2 != 0:
        raise ValueError("Number of mask coordinates is not even.")
    
    # Denormalize: convert normalized coordinates into pixel values.
    polygon_points = []
    for i in range(0, len(mask_coords_norm), 2):
        x_norm, y_norm = mask_coords_norm[i], mask_coords_norm[i+1]
        x = float(x_norm * image_width)
        y = float(y_norm * image_height)
        polygon_points.append((x, y))
    
    if not polygon_points:
        raise ValueError("No valid polygon points found.")
    
    # Create a Shapely polygon from the full-image polygon.
    poly_full = Polygon(polygon_points)
    if not poly_full.is_valid:
        poly_full = poly_full.buffer(0)
    
    # Define crop region using crop_box.
    x_offset, y_offset, crop_width, crop_height = crop_box
    crop_poly = box( x_offset, y_offset, x_offset + crop_width, y_offset + crop_height )
    
    # Compute intersection of polygon with crop.
    poly_crop = poly_full.intersection( crop_poly )
    if poly_crop.is_empty:
        return None  # no overlap with crop.
    
    # Handle different geometry types.
    if poly_crop.geom_type == "GeometryCollection":
        # Extract only polygonal components (Polygon or MultiPolygon)
        poly_candidates = [ geom for geom in poly_crop.geoms if geom.geom_type in [ "Polygon", "MultiPolygon" ] ]
        if not poly_candidates:
            return None  # No polygon components found.
        # If a MultiPolygon is among the candidates, flatten it.
        flattened = []
        for geom in poly_candidates:
            if geom.geom_type == "MultiPolygon":
                flattened.extend( list( geom.geoms ) )
            else:
                flattened.append( geom )
        # Select the largest polygon based on area.
        poly_crop = max( flattened, key = lambda p: p.area )
    elif poly_crop.geom_type == "MultiPolygon":
        poly_crop = max( poly_crop.geoms, key=lambda p: p.area )
    
    # Optionally, only keep detections where the intersection area is at least a fraction of the original.
    if poly_full.area > 0 and ( poly_crop.area / poly_full.area ) < area_thresh:
        return None

    # If max_points is provided, simplify the polygon.
    if max_points is not None:
        poly_crop = simplify_polygon_to_max_points(poly_crop, max_points)

    # Shift polygon coordinates to be relative to the crop.
    shifted_coords = []
    for x, y in np.array( poly_crop.exterior.coords ):
        shifted_coords.append( ( x - x_offset, y - y_offset ))
    
    new_poly = Polygon( shifted_coords) 
    if new_poly.is_empty:
        return None
    
    # Compute the bounding box of the shifted polygon.
    minx, miny, maxx, maxy = new_poly.bounds
    width = max( maxx - minx, 1 )
    height = max( maxy - miny, 1 )
    
    # Normalize bounding box coordinates with respect to the crop size. xleft, ytop, width, height
    new_bbox = ( minx / crop_width, miny / crop_height, width / crop_width, height / crop_height )
    new_bbox = convert_to_cxcywh( new_bbox ) # Convert to center x, center y, width, height
    
    # Convert polygon to NumPy array in OpenCV format.
    new_polygon_np = np.array( [ shifted_coords ], dtype = np.int32 )

    # Normalize polygon coordinates relative to the crop size.
    normalized_coords = [ ( x / crop_width, y / crop_height ) for x, y in shifted_coords ]
    new_polygon_np = np.array( [ normalized_coords ], dtype = np.float32 )

    # # === Create masks ===
    # masks = {}
    # for res in mask_shapes:

    #     mask = np.zeros( ( res, res ), dtype = np.int32 )
    #     cell_w = crop_width / res
    #     cell_h = crop_height / res

    #     for row in range( res ):
    #         for col in range( res ):
    #             cell_x1 = col * cell_w
    #             cell_y1 = row * cell_h
    #             cell_x2 = cell_x1 + cell_w
    #             cell_y2 = cell_y1 + cell_h
    #             cell_box = box( cell_x1, cell_y1, cell_x2, cell_y2 )

    #             if new_poly.intersects( cell_box ):
    #                 mask[row, col] += 1
    #     masks[res] = mask
        
    return label, new_bbox, new_polygon_np#, masks

def parse_annotation_line_for_crop_bbox(line, image_width, image_height, crop_box, area_thresh=0.5, max_points=None, mask_shapes=[]):
    
    # Split the annotation line and verify format.
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError("Annotation line must have exactly 5 values: label, center_x, center_y, w, h")
    
    # Parse label and bounding box (normalized).
    label = int( parts[0] )
    center_x_norm = float( parts[1] )
    center_y_norm = float( parts[2] )
    w_norm = float( parts[3] )
    h_norm = float( parts[4] )

    # Just create the poligon from the bounding box.
    half_w = w_norm / 2.0
    half_h = h_norm / 2.0
    top_left = ( center_x_norm - half_w, center_y_norm - half_h )
    top_right = ( center_x_norm + half_w, center_y_norm - half_h )
    bottom_right = ( center_x_norm + half_w, center_y_norm + half_h )
    bottom_left = ( center_x_norm - half_w, center_y_norm + half_h )
    
    # Compute the bounding box of the shifted polygon.
    new_bbox = ( center_x_norm, center_y_norm, w_norm, h_norm )    
    normalized_coords = [ top_left, top_right, bottom_right, bottom_left, top_left ]
    new_polygon_np = np.array( [ normalized_coords ], dtype = np.float32 )    
    
    return label, new_bbox, new_polygon_np  # masks

def merge_feature_maps( crop_boxes, crop_features, crop_size, image_size, fm_size):
    """
    Merge crop feature maps into a full feature map using torch.
    Overlapping regions are averaged.
    
    Parameters:
        crop_boxes (torch.Tensor): Tensor of shape [N, 4] with crop bounding boxes 
                                   (left, upper, right, lower) in original image coordinates.
        crop_features (torch.Tensor): Tensor of shape [N, C, fm_size, fm_size] containing the 
                                      feature maps for each crop.
        crop_size (int): The side length S of the original image crops (S x S).
        image_size (tuple): (width, height) of the original image.
        fm_size (int): The spatial dimension of the crop feature map (e.g., 64, 32, or 16).
    
    Returns:
        torch.Tensor: The merged full feature map of shape [C, full_height, full_width].
    """

    image_width, image_height = image_size
    
    # Scaling factor from image (crop) space to feature map space.
    factor = fm_size / crop_size

    # Compute full feature map dimensions.
    full_width = int( round( image_width * factor ) )
    full_height = int( round( image_height * factor ) )
    
    # Determine channels from crop_features (assumed shape [N, C, fm_size, fm_size]).
    N, C, s, s_check = crop_features.shape
    assert s == fm_size, "Feature map spatial size must match fm_size"

    device = crop_features.device
    dtype = crop_features.dtype

    # Create accumulators for features and weights.
    accumulator = torch.zeros( ( C, full_height, full_width ), dtype = dtype, device = device )
    weight = torch.zeros( ( full_height, full_width ), dtype = dtype, device = device )
    
    # Iterate over each crop.
    for i in range( len( crop_boxes ) ):

        # Retrieve crop box coordinates.
        left, upper, right, lower = crop_boxes[i]
        # Map the crop's top-left coordinates into full feature map space.
        dest_left = int( round( left * factor ) )
        dest_top  = int( round( upper * factor ) )
        # The region covered in the full feature map is fm_size x fm_size.
        dest_right = dest_left + fm_size
        dest_bottom = dest_top + fm_size

        # Accumulate the feature map.
        accumulator[ :, dest_top:dest_bottom, dest_left:dest_right ] += crop_features[i]
        # Track how many times each spatial location is updated.
        weight[ dest_top:dest_bottom, dest_left:dest_right ] += 1.0

    # Avoid division by zero.
    weight[weight == 0] = 1.0

    # Average the accumulated features.
    merged_feature_map = accumulator / weight.unsqueeze(0)

    return merged_feature_map

class TrainObjectsSegDataset(Dataset):

    def __init__(self, folder, transform=None, max_poly_points=None, crop_size=512, mode='train'):
        """
        Args:
            folder (str): Folder path containing images and labels.
            transform: Transformations to apply to images.
        """
        self.folder = folder
        self.transform = transform
        self.max_poly_points = max_poly_points
        self.mode = mode

        # Load classes from classes.txt.
        self.classes = []
        with open( os.path.join( self.folder, 'classes.txt' ), 'r' ) as f:
            for line in f:
                self.classes.append( line.strip() )

        # Insert no_object as 0 index.
        self.classes.insert( 0, 'no_object' )
        self.class_to_idx = { cls: idx for idx, cls in enumerate( self.classes ) }
        self.idx_to_class = { idx: cls for idx, cls in enumerate( self.classes ) }

        # List all image files from the images folder.
        # images_dir = os.path.join( self.folder, 'images' )
        images_dir = os.path.join( self.folder, f'images/{self.mode}' )
        self.filepaths = [ os.path.join( images_dir, f ) for f in os.listdir( images_dir ) if f.endswith( '.png' ) ]
        # Corresponding ground truth files (assumes same basename in labels folder).
        self.gt_files = [ f.replace( 'images', 'labels' ).replace( '.png', '.txt' ) for f in self.filepaths ]

        self.id_blacklist = []
        if os.path.exists( os.path.join( self.folder, 'blacklist.txt' ) ):
            with open( os.path.join( self.folder, 'blacklist.txt' ), 'r' ) as f:
                self.id_blacklist = [ int( line.strip() ) for line in f ]

        # Define crop size (fixed to 512×512 as requested).
        self.crop_size = crop_size

    def __len__(self):
        return len( self.filepaths )

    def __getitem__(self, idx):

        while idx in self.id_blacklist:
            idx = random.randint( 0, len( self.filepaths ) - 1 )
        # Retrieve sample by ID.
        sample_path = self.filepaths[idx]
        ann_path = self.gt_files[idx]

        if not os.path.exists( sample_path ):
            self.id_blacklist.append( idx )
            print( f"Image file not found for image {sample_path}. Skipping.")
            return self.__getitem__( idx )
        
        if not os.path.exists( ann_path ):
            self.id_blacklist.append( idx )
            print( f"No annotation file found for image {sample_path}. Skipping.")
            return self.__getitem__( idx )

        with open( ann_path, 'r' ) as f:
            lines = f.readlines()
        
        if len( lines ) == 0:
            self.id_blacklist.append( idx )
            print( f"No annotations found for image {sample_path}. Skipping.")
            return self.__getitem__( idx )
        
        # Load the image.
        image = decode( sample_path, "" ) # tensor
        _, orig_h, orig_w = image.shape

        # Random crop of size 512x512.
        max_x = orig_w - self.crop_size
        max_y = orig_h - self.crop_size
        x_offset = random.randint( 0, max_x )
        y_offset = random.randint( 0, max_y )
        crop_box = ( x_offset, y_offset, self.crop_size, self.crop_size )
        img_cropped = image[:, y_offset:y_offset+self.crop_size, x_offset:x_offset+self.crop_size]

        # Apply any additional transformations (e.g., converting to tensor).
        if self.transform:
            img_cropped = self.transform( img_cropped )

        annotations = []
        for line in lines:
            try:
                parsed = parse_seg_annotation_line_for_crop( line, 
                                                             orig_w, orig_h, 
                                                             crop_box, 
                                                             area_thresh = 0.1, 
                                                             max_points = self.max_poly_points - 1,
                                                             mask_shapes = [ 64, 32, 16 ] )
                if parsed is not None:
                    label, bbox, polygon_np = parsed
                    annotations.append({
                        "label": label,
                        "bbox": bbox,           # normalized bbox relative to crop
                        "polygon": polygon_np,   # polygon coordinates relative to crop
                    })
            except:
                pass
            
        if len( annotations ) == 0:
            annotations.append({
                "label": 24,
                "bbox": ( 0, 0, 0, 0 ),
                "polygon": np.zeros( ( 1, self.max_poly_points, 2 ), dtype = np.float32 ),
            })
                
        return img_cropped, annotations

    def check_bad_images(self):
        for idx in range( len( self ) ):
            self.__getitem__( idx )
        
        # save the blacklist
        with open( os.path.join( self.folder, 'blacklist.txt' ), 'w' ) as f:
            for item in self.id_blacklist:
                f.write( "%s\n" % item )

class TrainObjectsDetectDataset(Dataset):

    def __init__(self, folder, transform=None, max_poly_points=None, crop_size=512, mode='train'):
        """
        Args:
            folder (str): Folder path containing images and labels.
            transform: Transformations to apply to images.
        """
        self.folder = folder
        self.transform = transform
        self.max_poly_points = max_poly_points
        self.mode = mode

        # Load classes from classes.txt.
        self.classes = []
        with open( os.path.join( self.folder, 'classes.txt' ), 'r' ) as f:
            for line in f:
                self.classes.append( line.strip() )

        # Insert no_object as 0 index.
        self.classes.insert( 0, 'no_object' )
        self.class_to_idx = { cls: idx for idx, cls in enumerate( self.classes ) }
        self.idx_to_class = { idx: cls for idx, cls in enumerate( self.classes ) }

        # List all image files from the images folder.
        # images_dir = os.path.join( self.folder, 'images' )
        images_dir = os.path.join( self.folder, f'images/{self.mode}' )
        self.filepaths = [ os.path.join( images_dir, f ) for f in os.listdir( images_dir ) if f.endswith( '.png' ) ]
        # Corresponding ground truth files (assumes same basename in labels folder).
        self.gt_files = [ f.replace( 'images', 'labels' ).replace( '.png', '.txt' ) for f in self.filepaths ]

        self.id_blacklist = []
        if os.path.exists( os.path.join( self.folder, 'blacklist.txt' ) ):
            with open( os.path.join( self.folder, 'blacklist.txt' ), 'r' ) as f:
                self.id_blacklist = [ int( line.strip() ) for line in f ]

        # Define crop size (fixed to 512×512 as requested).
        self.crop_size = crop_size

    def __len__(self):
        return len( self.filepaths )

    def __getitem__(self, idx):

        while idx in self.id_blacklist:
            idx = random.randint( 0, len( self.filepaths ) - 1 )
        # Retrieve sample by ID.
        sample_path = self.filepaths[idx]
        ann_path = self.gt_files[idx]

        if not os.path.exists( sample_path ):
            self.id_blacklist.append( idx )
            print( f"Image file not found for image {sample_path}. Skipping.")
            return self.__getitem__( idx )
        
        if not os.path.exists( ann_path ):
            self.id_blacklist.append( idx )
            print( f"No annotation file found for image {sample_path}. Skipping.")
            return self.__getitem__( idx )

        with open( ann_path, 'r' ) as f:
            lines = f.readlines()
        
        if len( lines ) == 0:
            self.id_blacklist.append( idx )
            print( f"No annotations found for image {sample_path}. Skipping.")
            return self.__getitem__( idx )
        
        # Load the image.
        image = decode( sample_path, "" ) # tensor
        image = F.resize( image, ( 640, 640 ) )
        _, orig_h, orig_w = image.shape

        # Random crop of size 512x512.
        max_x = orig_w - self.crop_size
        max_y = orig_h - self.crop_size
        x_offset = random.randint( 0, max_x )
        y_offset = random.randint( 0, max_y )
        crop_box = ( x_offset, y_offset, self.crop_size, self.crop_size )
        img_cropped = image[:, y_offset:y_offset+self.crop_size, x_offset:x_offset+self.crop_size]

        # Apply any additional transformations (e.g., converting to tensor).
        if self.transform:
            img_cropped = self.transform( img_cropped )

        annotations = []
        for line in lines:
            try:
                parsed = parse_annotation_line_for_crop_bbox( line, 
                                                              orig_w, orig_h, 
                                                              crop_box, 
                                                              area_thresh = 0.1, 
                                                              max_points = self.max_poly_points - 1,
                                                              mask_shapes = [ 64, 32, 16 ] )
                if parsed is not None:
                    label, bbox, polygon_np = parsed
                    if label > len(self.classes) -2:
                        continue
                    annotations.append({
                        "label": label + 1,
                        "bbox": bbox,           # normalized bbox relative to crop
                        "polygon": polygon_np,   # polygon coordinates relative to crop
                        # "localization_mask": masks
                    })
            except:
                pass
            
        if len( annotations ) == 0:
            annotations.append({
                "label": 0,
                "bbox": ( 0, 0, 0, 0 ),
                "polygon": np.zeros( ( 1, 5, 2 ), dtype = np.float32 ),
                # "localization_mask": {
                #     64: np.zeros( ( 64, 64 ), dtype = np.int32 ),
                #     32: np.zeros( ( 32, 32 ), dtype = np.int32 ),
                #     16: np.zeros( ( 16, 16 ), dtype = np.int32 )
                # }
            })
                
        return img_cropped, annotations

    def check_bad_images(self):

        for idx in range( len( self ) ):
            self.__getitem__( idx )
        
        # save the blacklist
        with open( os.path.join( self.folder, 'blacklist.txt' ), 'w' ) as f:
            for item in self.id_blacklist:
                f.write( "%s\n" % item )

def filter_mask_by_size(mask, area_range):
  
    min_area, max_area = area_range
    mask = mask.astype(np.uint8)
    
    # Find contours in the mask.
    contours, _ = cv2.findContours( mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    
    # Start with an empty mask.
    filtered_mask = np.zeros_like( mask, dtype = np.uint8 )
    
    # Iterate over each contour, compute its area, and draw it if within range.
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bb_area = w * h
        if min_area <= bb_area < max_area:
            cv2.drawContours( filtered_mask, [cnt], -1, color = 1, thickness = -1 )
    
    return filtered_mask.astype(np.float32)

def collate_train_detection_data_and_cast(samples_list, mask_generator, mask_ratio_tuple, mask_probability, max_polygon_points):

    # Initialize lists for image and target data.
    images = []
    bounding_boxes = []
    polygons = []
    labels = []
    max_num_objects = np.max( [ len( sample[1] ) for sample in samples_list ] )
    # Iterate over the samples and extract image and target data.
    for sample in samples_list:
        image, targets = sample
        images.append( image )

        bounding_boxes_ = []
        polygons_ = []
        labels_ = []
        # Extract target data.
        for target in targets:

            label = target["label"]
            bbox = target["bbox"]
            polygon = target["polygon"]
            # localization_mask = target["localization_mask"]

            # Append target data to lists.
            labels_.append( label )
            bounding_boxes_.append( bbox )
            polygons_.append( polygon )

        if len( targets ) < max_num_objects:
            # Pad with zeros.
            num_pad = max_num_objects - len( targets )
            labels_.extend( [ 24 ] * num_pad )
            bounding_boxes_.extend( [ ( 0, 0, 0, 0 ) ] * num_pad )
            polygons_.extend( [ np.zeros( ( 1, max_polygon_points, 2 ), dtype = np.float32 ) ] * num_pad )

        labels.append( labels_ )
        bounding_boxes.append( bounding_boxes_ )
        polygons.append( np.concatenate( polygons_ ) )

    labels = np.array( labels, dtype = np.int32 )
    bounding_boxes = np.array( bounding_boxes, dtype = np.float32 )
    polygons = np.array( polygons, dtype = np.float32 )
    
    # Stack the lists to form tensors.
    images = torch.stack( images )
    labels = torch.tensor( labels, dtype = torch.long )
    bounding_boxes = torch.tensor( bounding_boxes, dtype = torch.float32 )
    polygons = torch.tensor( polygons, dtype = torch.float32 )

    B = len( images ) # Batch size
    N = 20*20
    n_samples_masked = int( B * mask_probability ) # Number of samples to be masked
    probs = torch.linspace( *mask_ratio_tuple, n_samples_masked + 1 ) # Linearly spaced probabilities
    upperbound = 0
    masks = [ ]
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks.append( torch.BoolTensor( mask_generator( int( N * random.uniform( prob_min, prob_max ) ) ) ) )
        upperbound += int( N * prob_max )
    for i in range( n_samples_masked, B ):
        masks.append( torch.BoolTensor( mask_generator(0) ) ) # No masking

    random.shuffle( masks )
    masks = torch.stack( masks ).flatten(1) # [ B, N ]
    indices = masks.flatten().nonzero().flatten() # [ B*N ]
    n_masked_patches = torch.full( (1,), fill_value = indices.shape[0], dtype = torch.long )
    masks_weight = ( 1 / masks.sum( -1 ).clamp( min = 1.0 ) ).unsqueeze(-1).expand_as( masks )[masks]

    return [ images, labels, bounding_boxes, polygons, ( masks, indices, n_masked_patches, masks_weight ) ]
    
def build_data_loader_train_detection(root, batch_size, max_objects, max_poly_points, crop_size, mode = 'detect', random=True, train=True):

    color_jittering = transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness = 0.4, contrast = 0.4, saturation = 0.2, hue = 0.1
                    )
                ],
                p = 0.8,
            ),
            transforms.RandomGrayscale( p = 0.2 ),
            GaussianBlur( p = 1.0 )
        ]
    )
    transform_color = transforms.Compose( [ color_jittering ] )
    # transform_color = transforms.Compose( [ color_jittering, make_normalize_transform() ] )
    # transform_color = transforms.Compose( [ make_normalize_transform() ] )

    if mode == 'detect':
        dataset = TrainObjectsDetectDataset( folder = root, transform = transform_color, max_poly_points = max_poly_points, crop_size = crop_size, mode = 'train' if train else 'val' )
    elif mode == 'seg':
        dataset = TrainObjectsSegDataset( folder = root, transform = transform_color, max_poly_points = max_poly_points, crop_size = crop_size, mode = 'train' if train else 'val' )
    
    # dataset.check_bad_images()
    mask_generator = MaskingGenerator( 20, num_masking_patches = 20*20 )

    collate_fn = partial(
        collate_train_detection_data_and_cast,
        mask_generator = mask_generator,
        mask_ratio_tuple = ( 0.15, 0.5 ),
        mask_probability = 0.5,
        max_polygon_points = max_poly_points
    )

    if random:

        sampler = InfiniteSampler( sample_count = len( dataset ), shuffle = True, seed = 37, advance = 1 )
        loader = DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = sampler,
            collate_fn = collate_fn,
            num_workers = 8, # High number of workers since CPU is not maxed
            persistent_workers = True,  # Keep workers alive
            pin_memory = True, # Faster CPU → GPU transfers
            prefetch_factor = 6, # Load more batches in advance
            timeout = 60,  # Prevent workers from resetting if disk is slow
            drop_last = True  # Avoid uneven batch issues
        )
    
    else:

        loader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = False,
            collate_fn = collate_fn,
            num_workers = 8, # High number of workers since CPU is not maxed
            persistent_workers = True,  # Keep workers alive
            pin_memory = True, # Faster CPU → GPU transfers
            prefetch_factor = 4, # Load more batches in advance
            timeout = 60,  # Prevent workers from resetting if disk is slow
            drop_last = True  # Avoid uneven batch issues
        )

    return loader

def extract_overlapping_crops_and_boxes(image: torch.Tensor, crop_size: int, stride: int):
    """
    Extract SxS crops from the image using a fixed stride, allowing control over overlap,
    and return the crop boxes.

    Parameters:
        image (torch.Tensor): The input image tensor of shape (C, H, W).
        crop_size (int): The side length S for the SxS crops.
        stride (int): The number of pixels to move for each crop. A value less than crop_size
                      results in overlapping crops.

    Returns:
        tuple: Two lists:
            - crops: a list of torch.Tensor crops sorted from top-left to bottom-right.
            - boxes: a list of tuples (left, upper, right, lower) for each crop.
    """
    _, height, width = image.shape
    crops = []
    boxes = []

    # Determine start positions along width and height.
    if width <= crop_size:
        left_positions = [0]
    else:
        left_positions = list(range(0, width - crop_size + 1, stride))
        if left_positions[-1] != width - crop_size:
            left_positions.append(width - crop_size)

    if height <= crop_size:
        upper_positions = [0]
    else:
        upper_positions = list(range(0, height - crop_size + 1, stride))
        if upper_positions[-1] != height - crop_size:
            upper_positions.append(height - crop_size)

    # Loop over the positions and extract crops.
    for upper in upper_positions:
        for left in left_positions:
            right = left + crop_size
            lower = upper + crop_size
            crop = image[:, upper:lower, left:right].clone()
            crops.append(crop)
            boxes.append((left, upper, right, lower))
    
    return crops, boxes

class InferenceObjectsDetectDataset(Dataset):

    def __init__(self, folder, transform=None, max_poly_points=None):
        """
        Args:
            folder (str): Folder path containing images and labels.
            transform: Transformations to apply to images.
        """
        self.folder = folder
        self.transform = transform
        self.max_poly_points = max_poly_points

        # Load classes from classes.txt.
        self.classes = []
        with open( os.path.join( self.folder, 'classes.txt' ), 'r' ) as f:
            for line in f:
                self.classes.append( line.strip() )

        # Insert no_object as 0 index.
        self.classes.insert( 0, 'no_object' )
        self.class_to_idx = { cls: idx for idx, cls in enumerate( self.classes ) }
        self.idx_to_class = { idx: cls for idx, cls in enumerate( self.classes ) }

        # List all image files from the images folder.
        # images_dir = os.path.join( self.folder, 'images' )
        images_dir = os.path.join( self.folder, 'images/train' )
        self.filepaths = [ os.path.join( images_dir, f ) for f in os.listdir( images_dir ) if f.endswith( '.png' ) ]
        # Corresponding ground truth files (assumes same basename in labels folder).
        self.gt_files = [ f.replace( 'images', 'labels' ).replace( '.png', '.txt' ) for f in self.filepaths ]

        self.id_blacklist = []

        # Define crop size (fixed to 512×512 as requested).
        self.crop_size = 512

    def __len__(self):
        return len( self.filepaths )

    def __getitem__(self, idx):

        while idx in self.id_blacklist:
            idx = random.randint( 0, len( self.filepaths ) - 1 )

        ann_path = self.gt_files[idx]
        if not os.path.exists( ann_path ):
            self.id_blacklist.append( idx )
            print( f"No annotation file found for image {sample_path}. Skipping.")
            return self.__getitem__( idx )

        with open( ann_path, 'r' ) as f:
            lines = f.readlines()
        
        if len( lines ) == 0:
            self.id_blacklist.append( idx )
            print( f"No annotations found for image {sample_path}. Skipping.")
            return self.__getitem__( idx )
        
        # Retrieve sample by ID.
        sample_path = self.filepaths[idx]

        # Load the image.
        img = decode( sample_path, "" ) # tensor
        _, orig_h, orig_w = img.shape

        crops, crop_boxes = extract_overlapping_crops_and_boxes( img, self.crop_size )
        data = []
        for img_cropped, crop_box in zip( crops, crop_boxes ):
        
            # Apply any additional transformations (e.g., converting to tensor).
            if self.transform:
                img_cropped = self.transform( img_cropped )

            # Load ground truth annotations.
            annotations = []
            for line in lines:
                try:
                    parsed = parse_seg_annotation_line_for_crop( line, 
                                                             orig_w, orig_h, 
                                                             crop_box, 
                                                             area_thresh = 0.1, 
                                                             max_points = self.max_poly_points )
                    if parsed is not None:
                        label, bbox, polygon_np = parsed
                        annotations.append({
                            "label": label,
                            "bbox": bbox,           # normalized bbox relative to crop
                            "polygon": polygon_np   # polygon coordinates relative to crop
                        })
                except Exception as e:
                    pass
            
            data.append( ( img_cropped, annotations, crop_box ) )
                
        return data

from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from PIL import ImageDraw

def draw(data, pred_boxes=None, grid_cols=4):
    """
    Randomly samples num_samples items from the dataset, draws bounding boxes and polygons,
    and returns (and optionally saves) a grid image of the results.
    
    Args:
        dataset (Dataset): Your DeficiencyDataset instance.
        num_samples (int): Total number of samples to draw (should equal grid_rows * grid_cols).
        grid_cols (int): Number of columns in the grid.
        output_path (str, optional): If provided, the grid image will be saved to this path.
        
    Returns:
        grid_img (PIL.Image): The resulting grid image with drawn annotations.
    """
    # We'll use ToPILImage to convert tensor images (if needed)
    to_pil = ToPILImage()
    to_tensor = ToTensor()

    if data[3] is None:
        data[3] = data[2]
    drawn_images = []
    size = len( data[0] )
    for i in range( size ):

        # img = denormalize_transform( data[0][i] )
        img = data[0][i]
        labels = data[1][i]
        bounding_boxes = data[2][i]
        polygons = data[3][i]
        pred_b = pred_boxes[i] if pred_boxes is not None else [None] * len( labels )
        
        # If the image is a tensor, convert it to a PIL image.
        if isinstance(img, torch.Tensor):
            # Assume image tensor is in [C,H,W] format.
            img = to_pil( img )
        
        # Create a drawing context.
        draw = ImageDraw.Draw( img )

        # The crop size is assumed to be the image size.
        crop_w, crop_h = img.size
        for label, bbox, polygon, p_b in zip( labels, bounding_boxes, polygons, pred_b ):

            if label == 0:
                continue
            
            # unormalized_coords = np.array( [ ( x * crop_w, y * crop_h ) for x, y in polygon ] )

            # Convert normalized bbox to absolute coordinates.
            x_center, y_center, width, height = bbox
            x_min = ( x_center - width / 2 ) * crop_w
            y_min = ( y_center - height / 2 ) * crop_h
            x_max = ( x_center + width / 2 ) * crop_w
            y_max = ( y_center + height / 2 ) * crop_h

            # Draw bounding box.
            draw.rectangle( [ x_min, y_min, x_max, y_max ], outline = "red", width = 2 )

            if p_b is not None:

                x_center, y_center, width, height = p_b
                x_min = ( x_center - width / 2 ) * crop_w
                y_min = ( y_center - height / 2 ) * crop_h
                x_max = ( x_center + width / 2 ) * crop_w
                y_max = ( y_center + height / 2 ) * crop_h

                draw.rectangle( [ x_min, y_min, x_max, y_max ], outline = "green", width = 2 )
                        
            # # Draw polygon if available.
            # if unormalized_coords is not None and unormalized_coords.shape[0] >= 3:
            #     # polygon_np is expected in OpenCV format: shape [1, num_points, 2].
            #     poly_points = [ tuple(pt) for pt in unormalized_coords ]
            #     draw.line( poly_points + [ poly_points[0] ], fill = "blue", width = 2 )

        # Append the drawn image.
        drawn_images.append(to_tensor(img))
    
    # Create a grid of images.
    grid = make_grid( drawn_images, nrow = grid_cols, padding = 4 )
    
    return grid

def nms_torch(bboxes, scores, iou_threshold=0.5):
    
    # Convert (c_x, c_y, w, h) to (x1, y1, x2, y2)
    half_w = bboxes[:, 2] / 2.0
    half_h = bboxes[:, 3] / 2.0
    x1 = bboxes[:, 0] - half_w
    y1 = bboxes[:, 1] - half_h
    x2 = bboxes[:, 0] + half_w
    y2 = bboxes[:, 1] + half_h

    # Stack them into a [num_detections, 4] tensor
    boxes_xyxy = torch.stack( [ x1, y1, x2, y2 ], dim = 1 )

    # Apply torchvision NMS
    keep_indices = nms( boxes_xyxy, scores, iou_threshold )
    return keep_indices