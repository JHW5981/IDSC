import argparse
import os
import sys
import math
import random
import loguru
import time
import json
from pathlib import Path

import numpy as np

from utils import *


def get_arguments():
    
    parser = argparse.ArgumentParser(description='IDSC arguments')
    
    # experiment
    parser.add_argument('--data_path', default="../../images/McM images/", help='path to the data')
    parser.add_argument('--output_directory', default='outputs', help='directory to save outputs')
    parser.add_argument("--patch_size", default=(16, 16), help="patch size and dictionary atom size")
    parser.add_argument("--patch_gap", type=int, default=14, help="Parameter that controls the overlapping of the patches")
    parser.add_argument("--num_atoms", type=int, default=256, help="the number of atoms")

    # loss
    parser.add_argument("--penalty_weight", type=float, default=50.0, help="the loss weight for sparse penalty term")
    parser.add_argument("--penalty_adjust_step", type=int, default=500, help="step for adjusting the loss weight for sparse penalty term")

    # train
    parser.add_argument("--num_epochs", type=int, default=100, help="the number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--log_steps", type=int, default=1000, help="logging steps")
 
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
    patches = get_patch_dataset(Path(args.data_path), args.patch_size, args.patch_gap)  # (8, 8, 3, patch_num)
    patch_num = patches.shape[-1]
    logger.log("INFO", f"Use {patch_num} patches for dictionary learning.")

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
    step = 0
    patch_indexes = [i for i in range(patch_num)]
    num_batches = math.floor(patch_num/args.batch_size)

    penalty_weight = args.penalty_weight
    
    for epoch in range(1, args.num_epochs+1):
        random.shuffle(patch_indexes)
        logger.log("INFO", f"Epoch {epoch}:")

        for i in range(num_batches):
            step += 1
            batch_patches = patches[:, patch_indexes[i*args.batch_size:(i+1)*args.batch_size]]
            batch_coefficients = coefficients[:, patch_indexes[i*args.batch_size:(i+1)*args.batch_size]]

            # update atoms
            L_atoms = np.linalg.norm(np.matmul(batch_coefficients, batch_coefficients.T)) + 0.1  # estimated Lipschitz constant
            gamma_D = 1.9 / L_atoms  # step size

            atoms_grad = - np.matmul(batch_patches - np.matmul(atoms, batch_coefficients), batch_coefficients.T)
            atoms = atoms - gamma_D * atoms_grad

            atoms = atoms / np.linalg.norm(atoms, axis=0)  # enforcing unit length on to the atoms
            
            # update coefficients
            L_coefficients = np.linalg.norm(np.matmul(atoms, atoms.T)) + 0.1  # estimated Lipschitz constant
            gamma_A = 1.9 / L_coefficients   # step size

            coefficients_grad = - np.matmul(atoms.T, batch_patches - np.matmul(atoms, batch_coefficients))
            coefficients_ = batch_coefficients - gamma_A * coefficients_grad
            
            # soft-thresholding shrinkage
            batch_coefficients = np.sign(coefficients_) * np.maximum(np.abs(coefficients_) - penalty_weight * gamma_A, 0)
            coefficients[:, patch_indexes[i*args.batch_size:(i+1)*args.batch_size]] = batch_coefficients
            
            if step % args.penalty_adjust_step == 0:
                penalty_weight = max(penalty_weight / 1.2, 1/2)
            
            if step % args.log_steps == 0:
                error = np.linalg.norm(patches - np.matmul(atoms, coefficients))
                logger.log("INFO", f"Epoch: {epoch}" + f"/{args.num_epochs}" + f", batch: {i+1}" + f"/{num_batches}" + f", error: {error:.6f}" + f", gamma A: {gamma_A:.3e}" + f", gamma D: {gamma_D:.3e}")
        
        error = np.linalg.norm(patches - np.matmul(atoms, coefficients))
        logger.log("INFO", f"Epoch {epoch} end." + f" Total error: {error:.6f}")

    logger.log("INFO", "Dictionary learning done.")
    
    # save dictionary 
    np.save(os.path.join(output_dir, 'atoms.npy'), atoms)
    atoms_save = atoms.reshape(*args.patch_size, 3, atoms.shape[-1])
    save_atom_image(atoms_save, os.path.join(output_dir, 'atoms.png'))
    

if __name__ == "__main__":
    main()

