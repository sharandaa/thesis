#!/bin/bash
#SBATCH --job-name=tfds
#SBATCH --output=/scratch/s2630575/logs/slurm/tfdstest2.out
#SBATCH --partition=graceADA

cd ..
python -u testimport.py> /scratch/s2630575/logs/python_output/tfdstest.txt