#!/bin/bash
#SBATCH --job-name=psnrplot
#SBATCH --output=/scratch/s2630575/logs/slurm/psnrplot.out
#SBATCH --partition=graceTST

python calculate_psnr.py