#!/bin/bash
#SBATCH --job-name=vit
#SBATCH --output=/scratch/s2630575/logs/slurm/vit.out
#SBATCH --partition=graceADA
#SBATCH --exclude = ethnode33
#SBATCH --exclude = ethnode34

cd ..
python -u vit.py> /scratch/s2630575/logs/python_output/vit.txt