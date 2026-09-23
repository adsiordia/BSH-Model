#!/bin/bash
#SBATCH --job-name=bsh_sweep50k
#SBATCH --output=logs/sweep_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=all

cd /home/adsiordia/BSH-Model-v2
export LD_PRELOAD=/home/adsiordia/miniconda3/lib/libstdc++.so.6
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== 4 embeddings x 4 algorithms, threshold 50,000 ==="
python -u src/sweep_embeddings.py
echo
echo "=== log loss curves ==="
python -u src/logloss_curves.py
echo
echo "=== methods export ==="
python -u src/export_methods.py
