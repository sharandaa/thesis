#!/bin/bash
#SBATCH --job-name=tfds
#SBATCH --output=/scratch/s2630575/logs/slurm/tfds2.out
#SBATCH --partition=graceTST

python AIDtestx2_dataset_builder.py