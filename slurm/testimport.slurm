#!/bin/bash
#SBATCH --job-name=tfds
#SBATCH --output=/scratch/s2630575/logs/slurm/aidtestx4.out
#SBATCH --partition=graceTST

cd ..
python -u testimport.py> /scratch/s2630575/logs/python_output/aidtestx4.txt