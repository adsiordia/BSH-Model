#!/bin/bash
#SBATCH --job-name=bsh_ft
#SBATCH --output=logs/ft_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD="$HOME/miniconda3/lib/libstdc++.so.6"
python -u src/analyze_taxa_activity.py 2>/dev/null
