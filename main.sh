#!/bin/bash

#SBATCH --job-name holoLDM_cosine_x0_holoLoss_1000_iter
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time 1-0
#SBATCH --partition batch_grad
#SBATCH -o slurm/logs/slurm-%A-%x.out

python main.py --epoch 1000 -b ./configs/hologram/holoLDM_cosine.yaml

exit 0