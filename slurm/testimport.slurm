#!/bin/bash
#SBATCH --job-name=tfdstest
#SBATCH --output=/scratch/s2630575/logs/slurm/tfdstest.out
#SBATCH --partition=graceTST

cd ..
python testimport.py