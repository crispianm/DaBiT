#!/bin/bash
#SBATCH --job-name=vk_depth
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/b5dh/DaBiT/logs/vk_depth_%j.out

cd /projects/b5dh/DaBiT
.venv/bin/python get_depths.py \
  -i /projects/b5dh/data/dabit_training_data/vkitti/frames \
  -o /projects/b5dh/data/dabit_training_data/vkitti-depth \
  -e vitl -d cuda
