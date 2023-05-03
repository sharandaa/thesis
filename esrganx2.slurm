#!/bin/bash
#SBATCH --job-name=esrgan_x2
#SBATCH --output=/scratch/s2630575/logs/slurm/esrganx2.out
#SBATCH --partition=graceTST

python -u inference_realesrgan.py> /scratch/s2630575/logs/python_output/realesrgan_x2.txt -n RealESRGAN_x2plus -i ../thesis/test_AID_x2 --outscale 2