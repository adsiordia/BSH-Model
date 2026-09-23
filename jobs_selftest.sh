#!/bin/bash
#SBATCH --job-name=bsh_selftest
#SBATCH --output=logs/selftest_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
cd /home/adsiordia/BSH-Model
export LD_PRELOAD=/home/adsiordia/miniconda3/lib/libstdc++.so.6
python -u src/selftest.py
