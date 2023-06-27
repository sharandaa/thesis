#!/bin/bash
#SBATCH --job-name=espcnft
#SBATCH --output=/scratch/s2630575/logs/slurm/espcnftpredict.out
#SBATCH --partition=graceTST

cd ..
python -u predict_ftespcn.py> /scratch/s2630575/logs/python_output/predictespcnft.txt