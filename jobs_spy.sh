#!/bin/bash
#SBATCH --job-name=bsh_spy
#SBATCH --output=logs/spy_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=all
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD="$HOME/miniconda3/lib/libstdc++.so.6"
python -u src/analyze_sp_yield.py 2>/dev/null
