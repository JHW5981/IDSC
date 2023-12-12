import os
from PIL import Image
import numpy as np
from typing import Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt

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
                   patch_size: Tuple[int, int]=(8, 8),
                   gap: int=4) -> np.ndarray:
    """
    Extract overlapping patches from the input image.

    Args:
        image (np.ndarray): Input image (height x width x channels).
        gap (int): Parameter that controls the overlapping.
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

    patches = []

    reS = round((height - patch_height) / gap + 1)
    I = np.linspace(0, height-patch_height, reS).round().astype(int)
    J = np.linspace(0, width-patch_width, reS).round().astype(int)

    for i in range(len(I)):
        for j in range(len(J)):
            patches.append(image[I[i]:I[i]+patch_height, J[j]:J[j]+patch_width, :])

    patches = np.stack(patches, axis=-1)
    
    return patches

def get_patch_dataset(data_path: Union[str, Path], 
                      image_resize:Tuple[int, int]=(512, 512), 
                      patch_size: Tuple[int, int]=(8, 8),
                      gap: int=4) -> np.ndarray:
    """
    Get all patches from the image data directory.

    Args:
        data_path (Union[str, Path]): Path to the image data directory, either as a string or a Path object.
        image_resize (Tuple[int, int]): Optional parameter to resize the loaded image. 
        patch_size (Tuple[int, int]): Size of each patch (patch_height x patch_width).
        gap (int): Parameter that controls the overlapping.
    
    Returns:
        np.ndarray: All extracted patches from the image data directory.
    
    Example:
        >>> from pathlib import Path
        >>> data_path = Path("path/to/data")
        >>> patches  = get_patch_dataset(data_path)  
    """
    file_paths = []
    for lists in os.listdir(data_path):
        path = os.path.join(data_path, lists)
        if path[-4:] == '.tif':
            file_paths.append(path)
    file_paths.sort()

    patches = []
    for path in file_paths:
        image_np = read_image(path, image_resize)  # (512, 512, 3)
        patch = extrat_patches(image_np, patch_size, gap)  # (8, 8, 3, patch_num)
        patches.append(patch)
    
    patches = np.concatenate(patches, axis=-1)

    return patches

def combine_patches(patches: np.ndarray, 
                    image_size: Tuple[int, int]=(512, 512),
                    gap: int=4) -> np.ndarray:
    """
    Combine all overlapping patches to reconstruct the full image.
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

    combined_image = np.zeros([height, width, channel])
    count_overlap = np.zeros([height, width, channel])

    reS = round((height - patch_height) / gap + 1)
    I = np.linspace(0, height-patch_height, reS).round().astype(int)
    J = np.linspace(0, width-patch_width, reS).round().astype(int)
    
    count = 0
    for i in range(len(I)):
        for j in range(len(J)):
            combined_image[I[i]:I[i]+patch_height, J[j]:J[j]+patch_width, :] += patches[:, :, :, count]
            count_overlap[I[i]:I[i]+patch_height, J[j]:J[j]+patch_width, :] += 1

            count += 1
    
    combined_image = combined_image / count_overlap

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

def save_atom_image(atom_np: np.ndarray,
                    save_path: Union[str, Path]) -> None:
    """
    Save atoms as an image to the specified path.

    Args:
        atom_np (np.ndarray): Learning atoms as type of numpy.ndarray.
        save_path (Union[str, Path]): Path to save the image file, either as a string or a Path object.

    Returns:
        none

    Examples:
        >>> from pathlib import Path
        >>> save_path = Path("path/to/save/image")
        >>> save_image(atoms.reshape(*args.patch_size, 3, atoms.shape[-1]), save_path)
    """
    patch_height, patch_width, channel, atom_num = atom_np.shape

    atom_min = np.min(atom_np.reshape(-1, channel, atom_num), axis=0)
    atom_max = np.max(atom_np.reshape(-1, channel, atom_num), axis=0)

    atom_np = (atom_np - atom_min) / (atom_max - atom_min)

    # get height num
    factors = []
    for i in range(1, atom_num+1):
        if atom_num % i == 0:
            factors.append(i)
    height_num = factors[len(factors)//2] 
    width_num = atom_num // height_num

    plt.figure(figsize=(0.5*width_num, 0.5*height_num))

    for i in range(atom_num):
        plt.subplot(height_num, width_num, i + 1)
        plt.imshow(atom_np[:, :, :, i])
        plt.xticks(())
        plt.yticks(())
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')


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