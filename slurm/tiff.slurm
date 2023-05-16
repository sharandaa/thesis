#!/bin/bash
#SBATCH --job-name=tiff
#SBATCH --output=/scratch/s2630575/logs/slurm/tiff.out
#SBATCH --partition=graceTST

cd ..
python converttiff.py