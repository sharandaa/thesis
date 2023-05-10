#!/bin/bash
#SBATCH --job-name=espcn
#SBATCH --output=/scratch/s2630575/logs/espcn.out
#SBATCH --partition=graceTST

python opencvmodel.py