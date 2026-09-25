#!/bin/bash
#SBATCH --job-name=bsh_hz
#SBATCH --output=logs/horizyn_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=all
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
/home/adsiordia/horizyn-env/bin/python -u src/horizyn_compare.py
