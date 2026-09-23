#!/bin/bash
#SBATCH --job-name=bsh_methods
#SBATCH --output=logs/methods_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=all
cd /home/adsiordia/BSH-Model-v2
export LD_PRELOAD=/home/adsiordia/miniconda3/lib/libstdc++.so.6
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u src/export_methods.py
