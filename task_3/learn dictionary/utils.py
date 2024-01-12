import os
from PIL import Image
import numpy as np
from typing import Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import h5py

def read_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Read an image from the specified path.

    Args:
        image_path (Union[str, Path]): Path to the image file, either as a string or a Path object.

    Returns:
        np.ndarray: Loaded image as type of numpy.ndarray.

    Examples:
        >>> from pathlib import Path
        >>> image_path = Path("path/to/image")
        >>> image_np = read_image(image_path) 
    """
    image = Image.open(image_path)
    image_np = np.array(image, dtype=np.float64)

    return image_np

def read_noise(noise_path: Union[str, Path]) -> np.ndarray:
    """
    Read the noise mat from the specified path.

    Args:
        noise_path (Union[str, Path]): Path to the noise file, either as a string or a Path object.

    Returns:
        np.ndarray: Loaded image as type of numpy.ndarray.

    Examples:
        >>> from pathlib import Path
        >>> noise_path = Path("path/to/noise")
        >>> noise_np = read_noise(noise_path) 
    """

    with h5py.File(noise_path, 'r') as f:
        noise = f['u_n'][:]
        noise_np = np.array(noise)
        noise_np = noise_np.transpose((2, 1, 0))

    return noise_np

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
        >>> patches  = extrat_patches(image_np, (8, 8), 4) 
    """
    height, width = image.shape[0], image.shape[1]
    patch_height, patch_width = patch_size

    patches = []

    reS = round((height - patch_height) / gap + 1)
    I = np.linspace(0, height-patch_height, reS).round().astype(int)
    J = np.linspace(0, width-patch_width, reS).round().astype(int)

    ndims = len(image.shape)
    for i in range(len(I)):
        for j in range(len(J)):
            patches.append(image[I[i]:I[i]+patch_height, J[j]:J[j]+patch_width] 
                           if ndims==2 else image[I[i]:I[i]+patch_height, J[j]:J[j]+patch_width, :])

    patches = np.stack(patches, axis=-1)
    
    return patches

def get_patch_dataset(data_path: Union[str, Path], 
                      patch_size: Tuple[int, int]=(8, 8),
                      gap: int=4) -> np.ndarray:
    """
    Get all patches from the image data directory.

    Args:
        data_path (Union[str, Path]): Path to the image data directory, either as a string or a Path object.
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
        image_np = read_image(path)  # (500, 500, 3)
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
    patch_height, patch_width = patches.shape[0], patches.shape[1]

    ndims = len(patches.shape) - 1

    combined_image = np.zeros([height, width]) if ndims==2 else np.zeros([height, width, patches.shape[2]])
    count_overlap = np.zeros([height, width]) if ndims==2 else np.zeros([height, width, patches.shape[2]])

    reS = round((height - patch_height) / gap + 1)
    I = np.linspace(0, height-patch_height, reS).round().astype(int)
    J = np.linspace(0, width-patch_width, reS).round().astype(int)
    
    count = 0
    for i in range(len(I)):
        for j in range(len(J)):
            if ndims == 2:
                # single channel
                combined_image[I[i]:I[i]+patch_height, J[j]:J[j]+patch_width] += patches[:, :, count]
                count_overlap[I[i]:I[i]+patch_height, J[j]:J[j]+patch_width] += 1
            else:
                # multi channel
                combined_image[I[i]:I[i]+patch_height, J[j]:J[j]+patch_width, :] += patches[:, :, :, count]
                count_overlap[I[i]:I[i]+patch_height, J[j]:J[j]+patch_width, :] += 1

            count += 1
    
    combined_image = combined_image / count_overlap

    return combined_image


def save_image(image_np: np.ndarray,
               save_path: Union[str, Path]) -> None:
    """
    Save an image to the specified path.

    Args:
        image_np (np.ndarray): Image as type of numpy.ndarray to be saved.
        save_path (Union[str, Path]): Path to save the image file, either as a string or a Path object.

    Returns:
        none

    Examples:
        >>> from pathlib import Path
        >>> save_path = Path("path/to/save/image")
        >>> save_image(image_np, save_path) 
    """

    image = Image.fromarray(image_np.astype(np.uint8))
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
    atom_num = atom_np.shape[-1]
    ndims = len(atom_np.shape) - 1

    atom_max = np.max(atom_np.reshape(-1, atom_num), axis=0)
    atom_min = np.min(atom_np.reshape(-1, atom_num), axis=0)
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
        if ndims == 2:
            # single channel
            plt.imshow(atom_np[:, :, i], cmap = plt.cm.gray, interpolation='nearest')
        else:
            # multi channel
            plt.imshow(atom_np[:, :, :, i])
        plt.xticks(())
        plt.yticks(())
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')


def evaluate_mse_and_psnr(original_np: np.ndarray, 
                          reconstruct_np: np.ndarray) -> Union[Tuple[float, float], Tuple[list, list]]:
    """
    Compute the MSE and PSNR of the reconstructed image.

    Args:
        original_np (np.ndarray): The original image numpy array.
        reconstruct_np (np.ndarray): The reconstructed image numpy array.

    Returns:
        Union[Tuple[float, float], Tuple[list, list]]: MSE, PSNR of the reconstructed image.

    Examples:
        >>> from pathlib import Path
        >>> mse, psnr = evaluate_psnr(original_np, reconstruct_np)
    """   

    assert original_np.shape == reconstruct_np.shape, 'Input images must have the same dimensions.'

    max_ = 255
    min_ = 0

    original_np = original_np.astype(dtype=np.float64)
    reconstruct_np = reconstruct_np.astype(dtype=np.float64)

    ndims = len(original_np.shape)
    if ndims == 2:
        # single channel
        mse = np.mean((original_np - reconstruct_np) ** 2, dtype=np.float64)
        psnr = 10 * np.log10(((max_ - min_) ** 2) / mse)

    else:
        # multi channel
        mse, psnr = [], []
        for i in range(original_np.shape[-1]):
            mse.append(np.mean((original_np[:,:,i] - reconstruct_np[:,:,i]) ** 2, dtype=np.float64))
            psnr.append(10 * np.log10(((max_ - min_) ** 2) / mse[i]))

    return mse, psnr

def init_atoms_coeff(patches: np.ndarray,
                     num_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the initial atoms and coefficients for dictionary learning.

    Args:
        pacthes (np.ndarray): the pacthes to be learned.

    Returns:
        Tuple[np.ndarray, np.ndarray]: initial atoms, initial coefficients

    Examples:
        >>> atoms, coefficients = init_atoms_coeff(patches, num_atoms)
    """   
    # do PCA to find principle component
    X = patches
    X_XT = np.matmul(X, X.T)

    eigenvalues, eigenvectors_X_XT = np.linalg.eig(X_XT)

    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors_X_XT = eigenvectors_X_XT[:, sort]

    atoms = eigenvectors_X_XT[:, 0:num_atoms].astype(np.float64)
    coefficients = np.matmul(atoms.T, patches).astype(np.float64)

    return atoms, coefficients