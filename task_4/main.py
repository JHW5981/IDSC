import argparse
import os
import sys
import loguru
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import openpyxl
from utils import *


def get_arguments():
    
    parser = argparse.ArgumentParser(description='IDSC arguments')
    
    # experiment
    parser.add_argument('--data_path', default="../images/McM images/McM18_noise.mat", help='path to the noise data')
    parser.add_argument('--clean_data_path', default="../images/McM images/McM18.tif", help='path to the clean data')
    parser.add_argument('--output_directory', default='outputs', help='directory to save outputs')
    parser.add_argument("--patch_size", nargs=2, type=int, default=(16, 16), help="patch size and dictionary atom size")
    parser.add_argument("--patch_gap", type=int, default=14, help="Parameter that controls the overlapping of the patches")
    parser.add_argument("--num_atoms", type=int, default=169, help="the number of atoms")

    # loss
    parser.add_argument("--penalty_weight", type=float, default=50.0, help="the loss weight for sparse penalty term")
    parser.add_argument("--penalty_adjust_step", type=int, default=500, help="step for adjusting the loss weight for sparse penalty term")

    # train
    parser.add_argument("--max_iters", type=int, default=2000, help="the max number of iterations")
    parser.add_argument("--log_steps", type=int, default=500, help="logging steps")
 
    # test
    parser.add_argument("--test_steps", type=int, default=500, help="testing steps")

    args = parser.parse_args()
    return args

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        assert os.path.isdir(path), f"'{path}' already exists but is not a directory."

def get_logger(path):
    logger = loguru.logger
    logger.remove()
    fmt_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level><n>{message}</n></level>"
    
    logger.add(sys.stderr, format=fmt_str, colorize=True, level="DEBUG")
    logger.info("Command executed: " + " ".join(sys.argv))

    fmt_str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    logger.add(path, format=fmt_str, level="INFO")
    logger.info(f"Logs are saved to {path}.")

    return logger


def main():
    args = get_arguments()

    # output path
    output_dir = os.path.join(args.output_directory, os.path.splitext(args.data_path.split('/')[-1])[0], time.strftime("%Y%m%d-%H%M%S"))
    # ensure_dir(output_dir)
    # logger
    # log_file = os.path.join(output_dir, "train.log")
    # logger = get_logger(log_file)
    # logger.log("INFO", "Configs:\n" + json.dumps(args.__dict__, indent=4))

    # read image
    image_np = read_image(Path(args.clean_data_path))  # (512, 512, (3,)) or (500, 500, (3,))
    noise_np = read_noise(Path(args.data_path))  # (512, 512, (3,)) or (500, 500, (3,))

    img_shape = (noise_np.shape[0], noise_np.shape[1])
    ndims = len(noise_np.shape)

    # extract patches from image
    patches = extrat_patches(noise_np, args.patch_size, args.patch_gap)  # (8, 8, (3,) patch_num)
    if ndims==2:
        patches = patches.reshape(-1, patches.shape[-1])  # (64, patch_num)
        patches_mean = np.mean(patches, axis=0)
        patches = patches - patches_mean
    else:
        patches = patches.reshape(-1, patches.shape[-2], patches.shape[-1])  # (64, 3, patch_num)
        patches_mean = np.mean(patches, axis=0)
        patches = patches - patches_mean
        patches = patches.reshape(-1, patches.shape[-1])  # (64*3, patch_num)
        
    # init atoms and coefficients 
    # U, sigma, VT = np.linalg.svd(patches)
    # sigma_mat = np.zeros((patches.shape[0], patches.shape[1]))
    # sigma_mat[:patches.shape[0], :patches.shape[0]] = np.diag(sigma)
    # atoms = U[:, 0:args.num_atoms]
    # coefficients = np.matmul(sigma_mat[0:args.num_atoms, :], VT)
    atoms, coefficients = init_atoms_coeff(patches, args.num_atoms)
    
    # dictionary learning
    penalty_weight = args.penalty_weight
    iters = []
    psnr_list = []
    mse_list = []

    for iter in range(1, args.max_iters):

        # update atoms
        L_atoms = np.linalg.norm(np.matmul(coefficients, coefficients.T)) + 0.1  # estimated Lipschitz constant
        gamma_D = 1.9 / L_atoms  # step size

        atoms_grad = - np.matmul(patches - np.matmul(atoms, coefficients), coefficients.T)
        atoms = atoms - gamma_D * atoms_grad

        atoms = atoms / np.linalg.norm(atoms, axis=0)  # enforcing unit length on to the atoms
        
        # update coefficients
        L_coefficients = np.linalg.norm(np.matmul(atoms, atoms.T)) + 0.1  # estimated Lipschitz constant
        gamma_A = 1.9 / L_coefficients   # step size

        coefficients_grad = - np.matmul(atoms.T, patches - np.matmul(atoms, coefficients))
        coefficients_ = coefficients - gamma_A * coefficients_grad
        
        # soft-thresholding shrinkage
        coefficients = np.sign(coefficients_) * np.maximum(np.abs(coefficients_) - penalty_weight * gamma_A, 0)
        
        if iter % args.penalty_adjust_step == 0:
            penalty_weight = max(penalty_weight / 1.5, 1/2)
        
        # if iter % args.log_steps == 0:
        #     error = np.linalg.norm(patches - np.matmul(atoms, coefficients))
        #     logger.log("INFO", f"iter: {iter}" + f"/{args.max_iters}" + f", error: {error:.6f}" + f", gamma A: {gamma_A:.3e}" + f", gamma D: {gamma_D:.3e}")

        if iter % args.test_steps == 0:
            # monitor training progress
            # reconstruct image from dictionary learning
            reconstruct_patches = np.matmul(atoms, coefficients)  # (64 or 192, patch_num)
            if ndims==2:
                reconstruct_patches = reconstruct_patches + patches_mean
                reconstruct_patches = reconstruct_patches.reshape(*args.patch_size, reconstruct_patches.shape[-1]) 
            else: 
                reconstruct_patches = reconstruct_patches.reshape(-1, 3, reconstruct_patches.shape[-1])
                reconstruct_patches = reconstruct_patches + patches_mean 
                reconstruct_patches = reconstruct_patches.reshape(*args.patch_size, 3, reconstruct_patches.shape[-1])  # (8, 8, (3,) patch_num)
            
            reconstruct_images = combine_patches(reconstruct_patches, img_shape, args.patch_gap)  # (512, 512, (3,)) or (500, 500, (3,))
            # evaluate mse and psnr
            mse, psnr = evaluate_mse_and_psnr(image_np, reconstruct_images)
            
            iters.append(iter)
            mse_list.append(mse)
            psnr_list.append(psnr)

    # logger.log("INFO", "Dictionary learning done.")
    
    # save dictionary and coefficients
    # np.save(os.path.join(output_dir, 'atoms.npy'), atoms)
    # atoms_save = atoms.reshape(*args.patch_size, atoms.shape[-1]) \
    #             if ndims==2 else atoms.reshape(*args.patch_size, 3, atoms.shape[-1])
    # save_atom_image(atoms_save, os.path.join(output_dir, 'atoms.png'))
    # np.save(os.path.join(output_dir, 'coefficients.npy'), coefficients)
    
    # reconstruct image from dictionary learning
    reconstruct_patches = np.matmul(atoms, coefficients)  # (64 or 192, patch_num)
    if ndims==2:
        reconstruct_patches = reconstruct_patches + patches_mean
        reconstruct_patches = reconstruct_patches.reshape(*args.patch_size, reconstruct_patches.shape[-1]) 
    else: 
        reconstruct_patches = reconstruct_patches.reshape(-1, 3, reconstruct_patches.shape[-1])
        reconstruct_patches = reconstruct_patches + patches_mean 
        reconstruct_patches = reconstruct_patches.reshape(*args.patch_size, 3, reconstruct_patches.shape[-1])  # (8, 8, (3,) patch_num)
    
    reconstruct_images = combine_patches(reconstruct_patches, img_shape, args.patch_gap)  # (512, 512, (3,)) or (500, 500, (3,))

    # save_image(reconstruct_images, os.path.join(output_dir, 'reconstruct.tif'))
    # np.save(os.path.join(output_dir, 'reconstruct.npy'), reconstruct_images)
    
    # evaluate mse and psnr
    mse, psnr = evaluate_mse_and_psnr(image_np, reconstruct_images)

    iters.append(iter)
    mse_list.append(mse)
    psnr_list.append(psnr)
    average_list = [np.mean(i) for i in psnr_list]
    max_psnr = max(average_list)
    max_index = average_list.index(max_psnr)

    # if ndims == 2:
    #     # single channel
    #     logger.log("INFO", f"BEST ITER: {iters[max_index]}")
    #     logger.log("INFO", f"BEST MSE: {mse_list[max_index]:.6f}" + f", BEST PSNR: {psnr_list[max_index]:.6f}")
    # else:
    #     # multiple channel
    #     logger.log("INFO", f"BEST ITER: {iters[max_index]}")
    #     logger.log("INFO", f"BEST MSE: ")
    #     logger.log("INFO", f"Red: {mse_list[max_index][0]:.6f}" + f", Green: {mse_list[max_index][1]:.6f}" + f", Blue: {mse_list[max_index][2]:.6f}")
    #     logger.log("INFO", f"Average: {np.mean(mse_list[max_index]):.6f}")

    #     logger.log("INFO", f"BEST PSNR: ")
    #     logger.log("INFO", f"Red: {psnr_list[max_index][0]:.6f}" + f", Green: {psnr_list[max_index][1]:.6f}" + f", Blue: {psnr_list[max_index][2]:.6f}")
    #     logger.log("INFO", f"Average: {np.mean(psnr_list[max_index]):.6f}")
    

    with open(f"./outputs/{os.path.splitext(args.data_path.split('/')[-1])[0]}/results.txt", "a+") as fp:
        fp.write(f"{os.path.basename(os.path.normpath(output_dir)):<20}{args.num_atoms:<15}{str(args.patch_size):<15}{args.patch_gap:<15}{args.penalty_weight:<15}{iters[max_index]:<15}{np.mean(psnr_list[max_index]):<15.6f}\n")


if __name__ == "__main__":
    main()
