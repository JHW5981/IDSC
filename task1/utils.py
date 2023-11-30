from PIL import Image
import numpy as np
from typing import Tuple, Union
from pathlib import Path

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error

def read_image(image_path: Union[str, Path], 
               resize:Tuple[int, int]=(512, 512)) -> np.ndarray:
    """
    Read an image from the specified path and normalized.

    Args:
        image_path (Union[str, Path]): Path to the image file, either as a string or a Path object.
        resize (Tuple[int, int]): Optional parameter to resize the loaded image. 

    Returns:
        np.ndarray: Loaded image as type of numpy.ndarray.

    Examples:
        >>> from pathlib import Path
        >>> image_path = Path("path/to/image")
        >>> image_np = read_image(image_path, (512, 512)) # Replace (512, 512) with your resize shape
    """
    image = Image.open(image_path)
    resized_image = image.resize(resize, Image.Resampling.BILINEAR)
    image_np = np.array(resized_image)

    image_np = image_np / 255.0  # do normalized

    return image_np

def extrat_patches(image: np.ndarray, 
                   patch_size: Tuple[int, int]=(8, 8)) -> np.ndarray:
    """
    Extract non-overlapping patches from the input image.

    Args:
        image (np.ndarray): Input image (height x width x channels).
        patch_size (Tuple[int, int]): Size of each patch (patch_height x patch_width).
    
    Returns:
        np.ndarray: Extracted patches with order of left to right, top to bottom.
    
    Example:
        >>> from PIL import Image
        >>> import numpy as np
        >>> image_np = np.array(Image.open("path/to/image")) 
        >>> patches  = extrat_patches(image_np, (8, 8))  # Replace (8, 8) with your actual patch size
    """
    height, width, channel = image.shape
    patch_height, patch_width = patch_size

    patches = image.reshape(height//patch_height, patch_height, width//patch_width, patch_width, channel)
    patches = np.transpose(patches, axes=(1, 3, 4, 0, 2)).reshape(*patch_size, channel, -1)
    
    return patches

def combine_patches(patches: np.ndarray, 
                    image_size: Tuple[int, int]=(512, 512)) -> np.ndarray:
    """
    Combine all patches to reconstruct the full image.
    This function is designed for validating the :func:`extract_patches` initially.

    Args:
        patches (np.ndarray): Patches extracted from :func:`extract_patches`.
        image_size (Tuple[int, int])): Size of the original image (height x width).

    Returns:
        np.ndarray: Reconstructed image.

    Example:
        >>> patches = extrat_patches(image_np) # from :func:`extrat_patched`
        >>> combined_image_np = combine_patches(patches, (512, 512)) # Replace (512, 512) with your actual image size
    """
    height, width = image_size
    patch_height, patch_width, channel, _ = patches.shape

    combined_image = patches.reshape(patch_height, patch_width, channel, height//patch_height, width//patch_width)
    combined_image = np.transpose(combined_image, axes=(3, 0, 4, 1, 2)).reshape(*image_size, channel)

    return combined_image


def save_image(image_np: np.ndarray,
               save_path: Union[str, Path], 
               resize: Tuple[int, int]=(500, 500)) -> None:
    """
    Save an image to the specified path.

    Args:
        image_np (np.ndarray): Image as type of numpy.ndarray to be saved.
        save_path (Union[str, Path]): Path to save the image file, either as a string or a Path object.
        resize (Tuple[int, int]): Optional parameter to resize the image. 

    Returns:
        none

    Examples:
        >>> from pathlib import Path
        >>> save_path = Path("path/to/save/image")
        >>> save_image(image_np, save_path, (500, 500)) # Replace (500, 500) with your resize shape
    """
    image_np = image_np * 255
    image = Image.fromarray(image_np.astype(np.uint8))
    image = image.resize(resize, resample=Image.Resampling.BILINEAR)
    image.save(save_path)

def evaluate_mse_and_psnr(original_path: Union[str, Path], 
                          reconstruct_path: Union[str, Path]) -> Tuple[float, float]:
    """
    Compute the MSE and PSNR of the reconstructed image.

    Args:
        original_path (Union[str, Path]): Path to the original image file, either as a string or a Path object.
        reconstruct_path (Union[str, Path]): Path to the reconstructed image file, either as a string or a Path object.

    Returns:
        Tuple[float, float]: MSE, PSNR of the reconstructed image.

    Examples:
        >>> from pathlib import Path
        >>> original_path = Path("path/to/original/image")
        >>> reconstruct_path = Path("path/to/reconstruct/image")
        >>> mse, psnr = evaluate_psnr(original_path, reconstruct_path)
    """
    original_np = np.array(Image.open(original_path))
    reconstruct_np = np.array(Image.open(reconstruct_path))    
    
    mse = mean_squared_error(reconstruct_np, original_np)
    psnr = peak_signal_noise_ratio(reconstruct_np, original_np)

    return mse, psnr