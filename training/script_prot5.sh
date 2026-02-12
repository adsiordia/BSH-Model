#!/bin/bash
#SBATCH --job-name=prot5_embeddings               # Job name
#SBATCH --ntasks=1
#SBATCH --output=my_job_output.log       # Output file
#SBATCH --gpus=1                     # Request 1 GPU
#SBATCH --cpus-per-task=20               # Request 20 CPUs
#SBATCH --mem=50G                       # Request 50 GB of RAM, note for Prot5 more memory might be needed

# Run your command
python prott5_embedder.py -i Seqs_list_total.fasta -o Seqs_list_total.h5 --per_protein 0               # Replace with your command
#         change with you files                     0: per-residue 1: per protein
