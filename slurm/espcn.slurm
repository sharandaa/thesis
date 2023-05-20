#!/bin/bash
#SBATCH --job-name=espcn
#SBATCH --output=/scratch/s2630575/logs/espcn.out
#SBATCH --partition=graceADA

python opencvmodel.py