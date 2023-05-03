#!/bin/bash
#SBATCH --job-name=psnr
#SBATCH --output=/scratch/s2630575/logs/slurm/psnr.out
#SBATCH --partition=graceTST

python -u calculate_psnr.py> /scratch/s2630575/logs/python_output/psnr.txt