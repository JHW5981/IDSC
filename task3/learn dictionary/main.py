import argparse
import os
import random
import sys
import loguru
import time
import math
import json
from pathlib import Path

import numpy as np

from utils import *
from grad import *


def get_arguments():
    
    parser = argparse.ArgumentParser(description='IDSC arguments')
    
    # experiment
    parser.add_argument('--data_path', default="../../images/origin/", help='path to the data')
    parser.add_argument('--output_directory', default='outputs', help='directory to save outputs')
    parser.add_argument("--original_img_shape", default=(500, 500), help="original image shape")
    parser.add_argument("--img_shape", default=(500, 500), help="image shape to resize to")
    parser.add_argument("--patch_size", default=(8, 8), help="patch size and dictionary atom size")
    parser.add_argument("--patch_gap", type=int, default=4, help="Parameter that controls the overlapping of the patches")
    parser.add_argument("--num_atoms", type=int, default=512, help="the number of atoms")

    # loss
    parser.add_argument('--penalty_type', type=str, default='huber', choices=['l1', 'huber'], help='function type for sparse penalty')
    parser.add_argument('--huber_M', type=float, default=0.5, help='M for Huber penalty function')
    parser.add_argument("--penalty_weight", type=float, default=10.0, help="the loss weight for sparse penalty term")

    # train
    parser.add_argument("--num_epochs", type=int, default=100, help="the number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--log_steps", type=int, default=1000, help="logging steps")

    # step size
    parser.add_argument('--step_size_type', type=str, default='bls', choices=['fixed', 'bls'], help='step size type, bls: backtracking line search')
    # fixed step size arguments
    parser.add_argument("--adjust_gamma_factor", type=float, default=0.95, help="multiply factor for fixed step size type")
    parser.add_argument("--adjust_gamma_steps", type=int, default=10000, help="update steps for fixed step size type")
    parser.add_argument("--gamma_A", type=float, default=1e-3, help="initial fixed step size of A")
    parser.add_argument("--gamma_D", type=float, default=5e-4, help="initial fixed step size of D")
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
    
    # get patch dataset
    patches = get_patch_dataset(Path(args.data_path), args.img_shape, args.patch_size, args.patch_gap)  # (8, 8, 3, patch_num)
    patch_num = patches.shape[-1]
    logger.log("INFO", f"Use {patch_num} patches for dictionary learning.")

    # random init atoms and coefficients 
    atoms = np.random.randn(*args.patch_size, 3, args.num_atoms)   # (8, 8, 3, 512)
    coefficients = np.random.randn(args.num_atoms, patch_num)   # (512, patch_num)
    
    # dictionary learning
    patches = patches.reshape(-1, patches.shape[-1])  # (192, patch_num)
    patches_mean = np.mean(patches, axis=0)
    patches_std = np.std(patches, axis=0)
    patches = (patches - patches_mean) / patches_std

    atoms = atoms.reshape(-1, atoms.shape[-1])   #(192, 512)
    
    step = 0
    patch_indexes = [i for i in range(patch_num)]
    num_batches = math.floor(patch_num/args.batch_size)
    
    if args.step_size_type == 'fixed':
        gamma_A = args.gamma_A
        gamma_D = args.gamma_D
    for epoch in range(1, args.num_epochs+1):
        random.shuffle(patch_indexes)
        logger.log("INFO", f"Epoch {epoch}:")

        for i in range(num_batches):
            step += 1
            batch_patches = patches[:, patch_indexes[i*args.batch_size:(i+1)*args.batch_size]]
            batch_coefficients = coefficients[:, patch_indexes[i*args.batch_size:(i+1)*args.batch_size]]

            # update coefficients
            coefficients_grad = calcu_grad_A(batch_patches, atoms, batch_coefficients, args.penalty_type, args.huber_M, args.penalty_weight)
            descent_direction_A = - coefficients_grad
            if args.step_size_type == 'bls':
                gamma_A = backtracking_ls_A(descent_direction_A, batch_patches, atoms, batch_coefficients, 
                                            args.penalty_type, args.huber_M, args.penalty_weight, args.alpha, args.beta_A)
            batch_coefficients = batch_coefficients + gamma_A * descent_direction_A
            coefficients[:, patch_indexes[i*args.batch_size:(i+1)*args.batch_size]] = batch_coefficients
            
            # update atoms
            atoms_grad = calcu_grad_D(batch_patches, atoms, batch_coefficients, args.penalty_weight/10.0)
            descent_direction_D = - atoms_grad
            if args.step_size_type == 'bls':
                gamma_D = backtracking_ls_D(descent_direction_D, batch_patches, atoms, batch_coefficients, 
                                            args.penalty_type, args.huber_M, args.penalty_weight, args.alpha, args.beta_D)
            atoms = atoms + gamma_D * descent_direction_D
            
            if (args.step_size_type == 'fixed') and (step % args.adjust_gamma_steps == 0):
                gamma_A = gamma_A * args.adjust_gamma_factor
                gamma_D = gamma_D * args.adjust_gamma_factor

            if step % args.log_steps == 0:
                error = np.mean(np.abs(batch_patches - np.matmul(atoms, batch_coefficients)))
                logger.log("INFO", f"Epoch: {epoch}" + f"/{args.num_epochs}" + f", batch: {i+1}" + f"/{num_batches}" + f", normalized error: {error:.6f}" + f", gamma A: {gamma_A:.3e}" + f", gamma D: {gamma_D:.3e}")
        
        error = np.mean(np.abs(patches - np.matmul(atoms, coefficients)))
        logger.log("INFO", f"Epoch {epoch} end." + f" Total normalized error: {error:.6f}")

    logger.log("INFO", "Dictionary learning done.")
    
    # save dictionary and coefficients
    np.save(os.path.join(output_dir, 'atoms.npy'), atoms)
    save_atom_image(atoms.reshape(*args.patch_size, 3, atoms.shape[-1]), os.path.join(output_dir, 'atoms.png'))
    np.save(os.path.join(output_dir, 'coefficients.npy'), coefficients)


if __name__ == "__main__":
    main()

