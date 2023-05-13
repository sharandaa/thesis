#!/bin/bash
#SBATCH --job-name=psnr
#SBATCH --output=/scratch/s2630575/logs/slurm/trainingcopy.out
#SBATCH --partition=graceTST

cd ..
python trainingcopy.py