#!/bin/bash
#SBATCH --job-name=resnetpredict
#SBATCH --output=/scratch/s2630575/logs/slurm/resnetpredict.out
#SBATCH --partition=graceTST

cd ..
python -u resnetpredict.py> /scratch/s2630575/logs/python_output/resnetpredictlightweight.txt