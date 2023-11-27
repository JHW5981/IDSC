from PIL import Image
import numpy as np
from typing import List, Tuple, Union
from pathlib import Path


def read_image(image_path: Union[str, Path], 
               resize:Tuple[int, int]=(512, 512)) -> np.ndarray:
    """
    Read an image from the specified path.

    Args:
        image_path (Union[str, Path]): Path to the image file, either as a string or a Path object.
        resize (Tuple[int, int]): Optional parameter to resize the loaded image. 

    Returns:
        np.ndarray: Loaded image as type of numpy.ndarray.

    Examples:
        >>> import pathlib as Path
        >>> image_path = Path("path/to/image")
        >>> image_np = read_image(image_path, (512, 512)) # Replace (512, 512) with your resize shape
    """
    image = Image.open(image_path)
    resized_image = image.resize(resize, Image.Resampling.BILINEAR)
    image_np = np.array(resized_image)

    return image_np

def extrat_patched(image: np.ndarray, 
                   patch_size: Tuple[int, int, int]=(8, 8, 3)) -> List[np.ndarray]:
    """
    Extract non-overlapping patches from the input image.

    Args:
        image (np.ndarray): Input image (height x width x channels).
        patch_size (Tuple[int, int, int]): Size of each patch (patch_height x patch_width x channels).
    
    Returns:
        List[np.ndarray]: List of extracted patches with order of left to right, top to bottom.
    
    Example:
        >>> from PIL import Image
        >>> import numpy as np
        >>> image_np = np.array(Image.open("path/to/image")) # Replace (8, 8, 3) with your actual patch size
        >>> patches  = extrat_patched(image_np, (8, 8, 3))
    """
    height, width, channel = image.shape
    patch_height, patch_width, patch_channel = patch_size

    patches = []

    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            patches.append(image[i:i+patch_height, j: j+patch_width, :])
    
    return patches

def combine_patches(patches: List[np.ndarray], 
                    image_size: Tuple[int, int, int]=(512, 512, 3)) -> np.ndarray:
    """
    Combine all patches to reconstruct the full image.
    This function is designed for validating the :func:`extract_patches` initially.

    Args:
        patches (List[np.ndarray]): Patches extracted from :func:`extract_patches`.
        image_size (Tuple[int, int, int])): Size of the original image (height x width x channels).

    Returns:
        np.ndarray: Reconstructed image.

    Example:
        >>> patches = extrat_patched(image_np) # from :func:`extrat_patched`
        >>> combined_image_np = extrat_patched(patches, (512, 512, 3)) # Replace (512, 512, 3) with your actual image size
    """
    height, width, channel = image_size
    patch_height, patch_width, patch_channel = patches[0].shape

    combined_image = np.zeros((height, width, channel), dtype=patches[0].dtype)

    idx = 0
    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            combined_image[i:i+patch_height, j:j+patch_width, :] = patches[idx]
            idx = idx + 1
    return combined_image
