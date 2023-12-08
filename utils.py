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
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np)) # normalization

    return image_np

def extrat_patches(image: np.ndarray,
                   gap: int=16,  
                   patch_size: Tuple[int, int]=(8, 8)) -> List[np.ndarray]:
    """
    Extract non-overlapping patches from the input image.

    Args:
        image (np.ndarray): Input image (height x width x channels(optional)).
        gap (int): Parameter that controls the overlapping, set to 0 with no overlapping
        patch_size (Tuple[int, int]): Size of each patch (patch_height x patch_width).
    
    Returns:
        List[np.ndarray]: List of extracted patches with order of left to right, top to bottom.
    
    Example:
        >>> from PIL import Image
        >>> import numpy as np
        >>> image_np = np.array(Image.open("path/to/image")) 
        >>> patches  = extrat_patches(image=image_np, gap=16, patch_size=(8, 8)) # Replace gap=16 and patch_size=(8, 8) with your actual needs
    """
    height, width = image.shape[0], image.shape[1]
    patch_height, patch_width = patch_size[0], patch_size[1]
    patches = []
    if gap==0:
        for i in range(0, height, patch_height):
            for j in range(0, width, patch_width):
                patches.append(image[i:i+patch_height, j: j+patch_width, :]\
                               if len(image.shape)==3 else image[i:i+patch_height, j: j+patch_width])
    else:
        reS = round(height/gap -5)
        I = np.linspace(0, height-patch_height, reS).round().astype(int)
        J = np.linspace(0, width-patch_width, reS).round().astype(int)
        for i in range(len(I)):
            for j in range(len(J)):
                patches.append(image[I[i]:I[i]+patch_height, J[j]:J[j]+patch_width, :]\
                               if len(image.shape)==3 else image[I[i]:I[i]+patch_height, J[j]:J[j]+patch_width])
    return patches

def combine_patches(patches: List[np.ndarray], 
                    image_size: Tuple[int, int]=(512, 512)) -> np.ndarray:
    """
    Combine all patches to reconstruct the full image. TODO: not implemented for gap ≥ 1
    This function is designed for validating the :func:`extract_patches` initially.

    Args:
        patches (List[np.ndarray]): Patches extracted from :func:`extract_patches`.
        image_size (Tuple[int, int])): Size of the original image (height x width).

    Returns:
        np.ndarray: Reconstructed image (height x width x channels(optional)).

    Example:
        >>> patches = extrat_patched(image_np) # from :func:`extrat_patched`
        >>> combined_image_np = combine_patches(patches, (512, 512)) # Replace (512, 512) with your actual needs
    """
    height, width = image_size[0], image_size[1]
    patch_height, patch_width = patches[0].shape[0], patches[0].shape[1]
    if len(patches[0].shape) == 3:
        channel = 3
    else:
        channel = None

    combined_image = np.zeros((height, width, channel), dtype=patches[0].dtype) if channel else np.zeros((height, width), dtype=patches[0].dtype)

    idx = 0
    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            if channel:
                combined_image[i:i+patch_height, j:j+patch_width, :] = patches[idx]
            else:
                combined_image[i:i+patch_height, j:j+patch_width] = patches[idx]
            idx = idx + 1
    return combined_image

def concat_patches(patches:List[np.ndarray]) -> np.ndarray:
    """
    Concatenate a list of patches into a big 3-D array.
    The length of the list `patches` is N, each patch with shape (8, 8, 3) by default.
    The shape of concatenated 3-D array is then (8x8, N, 3).
    Each column of the 2-D array is formed by concatenating the columns of each patch.

    Args:
        patches (List[np.ndarray]): Input list of patch arrays.
    
    Returns:
        np.ndarray: A 3-D array by concatenating the patches.
    
    Example:
        >>> image_np = read_image(image_path)
        >>> patches = extrat_patched(image_np)
        >>> concatenated_patches = concat_patches(patches)
    """
    reshaped_patches = [np.transpose(patch, axes=(1, 0, 2)).copy().reshape(-1, 3) if len(patch.shape)==3 else np.transpose(patch, axes=(1, 0)).copy().reshape(-1) for patch in patches]
    concatenated_patches = np.stack(reshaped_patches, axis=1)

    return concatenated_patches

def deconcat_patches(concatenated_patches:np.ndarray, 
                     patch_size: Tuple[int, int]=(8, 8)) -> List[np.ndarray]:
    """
    Deconcatenate a concatenated_patchesarray into a list of patches.

    Args:
        concatenated_patches (np.ndarray): Input array that is concatenated.
        patch_size (Tuple[int, int]): Patch size that we want to decompose to.
    
    Returns:
        List[np.ndarray]: List of patches after deconcatenation.
    
    Example:
        >>> image_np = read_image(image_path)
        >>> patches = extrat_patches(image_np)
        >>> concatenated_patches = concat_patches(patches)
        >>> pathes_restore = deconcat_patches(concatenated_patches) # pathes_restore == patches √
    """
    concatenated_patches_copy = concatenated_patches.copy()
    N = concatenated_patches_copy.shape[1]
    patches_list = []
    for i in range(N):
        patch = concatenated_patches_copy[:, i, :].reshape((concatenated_patches_copy.shape[0],3)).reshape((patch_size[0],patch_size[1],3)).transpose((1,0,2)) if len(concatenated_patches_copy.shape)==3 \
            else concatenated_patches_copy[:, i].reshape((concatenated_patches_copy.shape[0])).reshape((patch_size[0],patch_size[1])).transpose((1,0))
        
        patches_list.append(patch)

    return patches_list