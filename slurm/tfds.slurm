#!/bin/bash
#SBATCH --job-name=tfds
#SBATCH --output=/scratch/s2630575/logs/slurm/tfds.out
#SBATCH --partition=graceADA

python testAIDx2.py