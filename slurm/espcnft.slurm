#!/bin/bash
#SBATCH --job-name=espcnft
#SBATCH --output=/home/s2630575/logs/slurm/espcnft.out
#SBATCH --partition=graceADA

cd ..
python -u finetune_espcn.py> /home/s2630575/logs/python_output/espcnft.txt