#!/bin/bash
#SBATCH --job-name=tfds
#SBATCH --output=/scratch/s2630575/logs/slurm/aidtestx4.out
#SBATCH --partition=graceTST

tfds build --data_dir='/scratch/s2630575/tfds' --manual_dir='/scratch/s2630575/datasets' aidtestx4.py