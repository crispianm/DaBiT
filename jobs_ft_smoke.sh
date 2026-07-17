#!/bin/bash
#SBATCH --job-name=dabit_ft_smoke
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/projects/b5dh/DaBiT/logs/dabit_smoke_%j.out
cd /projects/b5dh/DaBiT
.venv/bin/python train.py -c configs/dabit_ft_smoke.json
