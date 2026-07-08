#!/bin/bash
#SBATCH --job-name=dabit_ddp_smoke
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=00:45:00
#SBATCH --output=/projects/b5dh/DaBiT/logs/dabit_ddp_smoke_%j.out

cd /projects/b5dh/DaBiT
( sleep 300; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader ) &
.venv/bin/torchrun --standalone --nproc_per_node=4 train.py -c configs/dabit_ddp_smoke.json
