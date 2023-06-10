#!/bin/bash
#SBATCH --job-name=resnet
#SBATCH --output=/scratch/s2630575/logs/slurm/resnet.out
#SBATCH --partition=graceADA

cd ..
python -u resnet.py> /scratch/s2630575/logs/python_output/resnet36test.txt