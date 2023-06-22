#!/bin/bash
#SBATCH --job-name=espcnft
#SBATCH --output=/scratch/s2630575/logs/slurm/espcnftx3_39.out
#SBATCH --partition=graceADA

cd ..
python -u finetune_espcn.py> /scratch/s2630575/logs/python_output/espcnftx3_39.txt