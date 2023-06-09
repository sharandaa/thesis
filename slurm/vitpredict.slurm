#!/bin/bash
#SBATCH --job-name=vitpredict
#SBATCH --output=/scratch/s2630575/logs/slurm/vitpredict.out
#SBATCH --partition=graceTST

cd ..
python -u vitpredict.py> /scratch/s2630575/logs/python_output/vitpredict35.txt