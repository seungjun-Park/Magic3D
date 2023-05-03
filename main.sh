#!/bin/bash

#SBATCH --job-name Magic3D_save_cat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time 1-0
#SBATCH --partition batch_grad
#SBATCH -o slurm/logs/slurm-%A-%x.out

python main.py  --workspace trial4 -O --test --save_mesh

exit 0