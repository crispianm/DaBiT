#!/bin/bash
#SBATCH --job-name=vos_depth
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/b5dh/DaBiT/logs/vos_depth_%A_%a.out

cd /projects/b5dh/DaBiT
.venv/bin/python get_depths.py \
  -i /projects/b5dh/data/dabit_training_data/vos_shards/shard_${SLURM_ARRAY_TASK_ID} \
  -o /projects/b5dh/data/dabit_training_data/youtube-vos-depth \
  -e vitl -d cuda
