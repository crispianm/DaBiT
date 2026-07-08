#!/bin/bash
#SBATCH --job-name=dabit_ddp
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/b5dh/DaBiT/logs/dabit_ddp_%j.out

cd /projects/b5dh/DaBiT
# Trainer auto-resumes from <out_dir>/model_latest.pth; re-submission continues the run.
.venv/bin/torchrun --standalone --nproc_per_node=4 train.py -c configs/dabit_ddp.json
