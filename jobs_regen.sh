#!/bin/bash
#SBATCH --job-name=bsh_regen50k
#SBATCH --output=logs/regen_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=all

cd /home/adsiordia/BSH-Model-v2
export LD_PRELOAD=/home/adsiordia/miniconda3/lib/libstdc++.so.6
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# candidate predictions need the retrained model, so they come first
echo "=== candidate predictions (66 signal-peptide sequences) ==="
python -u src/predict_candidates.py
echo
for s in export_amine_chem export_trends export_data_tab export_limits \
         export_candidates export_trimming export_methods; do
  echo "=== $s ==="
  python -u src/$s.py
  echo
done
