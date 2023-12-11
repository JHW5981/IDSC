import argparse
import os
import sys
import loguru
import time
import json
from pathlib import Path

import numpy as np

from utils import *
from grad import *


def get_arguments():
    
    parser = argparse.ArgumentParser(description='IDSC arguments')
    
    # experiment
    parser.add_argument('--data_path', default="../images/origin/McM01.tif", help='path to the data')
    parser.add_argument('--output_directory', default='outputs', help='directory to save outputs')
    parser.add_argument("--original_img_shape", default=(500, 500), help="original image shape")
    parser.add_argument("--img_shape", default=(512, 512), help="image shape to resize to")
    parser.add_argument("--patch_size", default=(8, 8), help="patch size and dictionary atom size")
    parser.add_argument("--patch_gap", type=int, default=6, help="Parameter that controls the overlapping of the patches")
    parser.add_argument("--num_atoms", type=int, default=128, help="the number of atoms")

    # loss
    parser.add_argument('--penalty_type', type=str, default='huber', choices=['l1', 'huber'], help='function type for sparse penalty')
    parser.add_argument('--huber_M', type=float, default=0.5, help='M for Huber penalty function')
    parser.add_argument("--penalty_weight", type=float, default=10.0, help="the loss weight for sparse penalty term")

    # train
    parser.add_argument("--max_iters", type=int, default=800, help="the number of iterations")
    parser.add_argument("--log_steps", type=int, default=10, help="logging steps")

    # step size
    parser.add_argument('--step_size_type', type=str, default='bls', choices=['fixed', 'bls'], help='step size type, bls: backtracking line search')
    # fixed step size arguments
    parser.add_argument("--adjust_gamma_factor", type=float, default=0.95, help="multiply factor for fixed step size type")
    parser.add_argument("--adjust_gamma_steps", type=int, default=1000, help="update steps for fixed step size type")
    parser.add_argument("--gamma_A", type=float, default=1e-4, help="initial fixed step size of A")
    parser.add_argument("--gamma_D", type=float, default=1e-3, help="initial fixed step size of D")
    # backtracking line search arguments
    parser.add_argument("--alpha", type=float, default=0.3, help="alpha in backtracking line search")
    parser.add_argument("--beta_D", type=float, default=0.25, help="beta for D in backtracking line search")
    parser.add_argument("--beta_A", type=float, default=0.25, help="beta for A in backtracking line search")
 
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
    ensure_dir(output_dir)
    # logger
    log_file = os.path.join(output_dir, "train.log")
    logger = get_logger(log_file)
    logger.log("INFO", "Configs:\n" + json.dumps(args.__dict__, indent=4))
    
    # read image
    image_np = read_image(Path(args.data_path), args.img_shape)  # (512, 512, 3)

    # extract patches from image
    patches = extrat_patches(image_np, args.patch_size, args.patch_gap)  # (8, 8, 3, patch_num)

    # random init atoms and coefficients 
    atoms = np.random.randn(*args.patch_size, 3, args.num_atoms)   # (8, 8, 3, 128)
    coefficients = np.random.randn(args.num_atoms, patches.shape[-1])   # (128, patch_num)
    
    # dictionary learning
    patches = patches.reshape(-1, patches.shape[-1])  # (192, patch_num)
    patches_mean = np.mean(patches, axis=0)
    patches_std = np.std(patches, axis=0)
    patches = (patches - patches_mean) / patches_std

    atoms = atoms.reshape(-1, atoms.shape[-1])   #(192, 128)
    
    if args.step_size_type == 'fixed':
        gamma_A = args.gamma_A
        gamma_D = args.gamma_D
    for iter in range(1, args.max_iters):
        # update coefficients
        coefficients_grad = calcu_grad_A(patches, atoms, coefficients, args.penalty_type, args.huber_M, args.penalty_weight)
        descent_direction_A = - coefficients_grad
        if args.step_size_type == 'bls':
            gamma_A = backtracking_ls_A(descent_direction_A, patches, atoms, coefficients, 
                                        args.penalty_type, args.huber_M, args.penalty_weight, args.alpha, args.beta_A)
        coefficients = coefficients + gamma_A * descent_direction_A
        
        # update atoms
        atoms_grad = calcu_grad_D(patches, atoms, coefficients, args.penalty_weight/10.0)
        descent_direction_D = - atoms_grad
        if args.step_size_type == 'bls':
            gamma_D = backtracking_ls_D(descent_direction_D, patches, atoms, coefficients, 
                                        args.penalty_type, args.huber_M, args.penalty_weight, args.alpha, args.beta_D)
        atoms = atoms + gamma_D * descent_direction_D
        
        
        if (args.step_size_type == 'fixed') and (iter % args.adjust_gamma_steps == 0):
            gamma_A = gamma_A * args.adjust_gamma_factor
            gamma_D = gamma_D * args.adjust_gamma_factor

        if iter % args.log_steps == 0:
            error = np.mean(np.abs(patches - np.matmul(atoms, coefficients)))
            logger.log("INFO", f"iter: {iter}" + f"/{args.max_iters}" + f", normalized error: {error:.6f}" + f", gamma A: {gamma_A:.3e}" + f", gamma D: {gamma_D:.3e}")
    logger.log("INFO", "Dictionary learning done.")
    
    # save dictionary and coefficients
    np.save(os.path.join(output_dir, 'atoms.npy'), atoms)
    save_atom_image(atoms.reshape(*args.patch_size, 3, atoms.shape[-1]), os.path.join(output_dir, 'atoms.png'))
    np.save(os.path.join(output_dir, 'coefficients.npy'), coefficients)
    
    # reconstruct image from dictionary learning
    reconstruct_patches = np.matmul(atoms, coefficients)  # (192, patch_num)
    reconstruct_patches = reconstruct_patches * patches_std + patches_mean
    reconstruct_patches = reconstruct_patches.reshape(*args.patch_size, 3, reconstruct_patches.shape[-1]) # (8, 8, 3, patch_num)
    reconstruct_images = combine_patches(reconstruct_patches, args.img_shape, args.patch_gap)  # (512, 512, 3)

    save_image(reconstruct_images, 
               os.path.join(output_dir, 'reconstruct.tif'),
               args.original_img_shape)
    
    # evaluate mse and psnr
    mse, psnr = evaluate_mse_and_psnr(Path(args.data_path), os.path.join(output_dir, 'reconstruct.tif'))
    logger.log("INFO", f"MSE: {mse:.6f}" + f", PSNR: {psnr:.6f}")

if __name__ == "__main__":
    main()

