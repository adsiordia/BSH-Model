#!/bin/bash
#SBATCH --job-name=bsh_retrain50k
#SBATCH --output=logs/retrain_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=all

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD="$HOME/miniconda3/lib/libstdc++.so.6"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== production model, threshold 50,000 ==="
python -u src/train_production.py
echo
echo "=== site predictions (out-of-fold) ==="
python -u src/export_site_data.py
