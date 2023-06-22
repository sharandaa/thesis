#!/bin/bash
#SBATCH --job-name=psnr
#SBATCH --output=/home/s2630575/logs/slurm/psnr.out
#SBATCH --partition=graceTST

python -u calculate_psnr.py> /home/s2630575/logs/python_output/espcnx3.txt