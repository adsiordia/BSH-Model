#!/bin/bash
#SBATCH --job-name=bsh_fixdata
#SBATCH --output=logs/fixdata_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=all
cd /home/adsiordia/BSH-Model-v2
export LD_PRELOAD=/home/adsiordia/miniconda3/lib/libstdc++.so.6
python -u src/export_data_tab.py
python -u src/export_limits.py
