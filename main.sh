#!/bin/bash

#SBATCH --job-name dreamfusion_hamburger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time 1-0
#SBATCH --partition batch_grad
#SBATCH -o slurm/logs/slurm-%A-%x.out

python main.py --text "a hamburger" --workspace trial -O

exit 0