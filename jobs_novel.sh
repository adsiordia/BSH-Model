#!/bin/bash
#SBATCH --job-name=bsh_novel
#SBATCH --output=logs/novel_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
cd /home/adsiordia/BSH-Model-v2
export LD_PRELOAD=/home/adsiordia/miniconda3/lib/libstdc++.so.6
python -u src/analyze_novel.py
