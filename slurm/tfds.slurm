#!/bin/bash
#SBATCH --job-name=tfds
#SBATCH --output=/scratch/s2630575/logs/slurm/aidtestx3.out
#SBATCH --partition=graceTST

python aidtestx3.py