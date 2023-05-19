#!/bin/bash
#SBATCH --job-name=label
#SBATCH --output=/scratch/s2630575/logs/slurm/label.out
#SBATCH --partition=graceTST

cd ..
python -u labeldataframe.py> /scratch/s2630575/logs/python_output/label.txt