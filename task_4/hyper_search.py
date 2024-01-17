from sklearn.model_selection import ParameterGrid
import subprocess
from tqdm import tqdm

# get our best results
lst = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18']
for i in tqdm(lst):
    with open(f"./outputs/McM{i}_noise/results.txt", "a+") as fp:
        fp.write("===========================================get our best results===========================================\n")
        fp.write("{: <20}{: <15}{: <15}{: <15}{: <15}{: <15}{: <15}\n".format(
            "dirname", "num_atoms", "patch_size", "patch_gap", "penalty_weight", "best_iter", "psnr"))

    param_grid = {'patch_size': [(16, 16)], 'num_atoms': [256], 'patch_gap': [1]}
    grid = list(ParameterGrid(param_grid))
    for para in grid:
        command = f"python main.py --data_path \"../images/McM images/McM{i}_noise.mat\" --clean_data_path \"../images/McM images/McM{i}.tif\" --patch_size {para['patch_size'][0]} {para['patch_size'][1]} --patch_gap {para['patch_gap']} --num_atoms 256 --penalty_weight 20 --max_iters 1500  --test_steps 100"
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
