#!/bin/bash
#SBATCH --job-name=tfds
#SBATCH --output=/scratch/s2630575/logs/slurm/tfdsaidx2.out
#SBATCH --partition=graceTST

python aidx2.py