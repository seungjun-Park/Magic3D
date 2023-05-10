#!/bin/bash

#SBATCH --job-name Magic3D_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time 1-0
#SBATCH --partition batch_grad
#SBATCH -o slurm/logs/slurm-%A-%x.out

python main.py --text "anime, masterpiece, high quality, 1girl, solo, long hair, looking at viewer, blush, smile, bangs, blue eyes, skirt, medium breasts, iridescent, gradient, colorful, besides a cottage, in the country" --workspace logs/first_stage/test2 -O --hf_key dreamlike-art/dreamlike-anime-1.0

exit 0