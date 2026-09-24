#!/bin/bash
#SBATCH --job-name=bsh_trimtruth
#SBATCH --output=logs/trimtruth_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=all
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD="$HOME/miniconda3/lib/libstdc++.so.6"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u src/analyze_trim_truth.py
