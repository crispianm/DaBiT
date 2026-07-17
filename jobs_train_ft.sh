#!/bin/bash
#SBATCH --job-name=dabit_ft
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/b5dh/DaBiT/logs/dabit_train_%j.out

cd /projects/b5dh/DaBiT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Trainer auto-resumes from <out_dir>/model_latest.pth, so re-submitting this
# script continues an interrupted run.
.venv/bin/python train.py -c configs/dabit_ft.json
