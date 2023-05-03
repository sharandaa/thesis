#!/bin/bash
#SBATCH --job-name=esrgan_x4
#SBATCH --output=/scratch/s2630575/logs/slurm/esrganx4.out
#SBATCH --partition=graceTST

python -u inference_realesrgan.py> /scratch/s2630575/logs/python_output/realesrgan_x4.txt -n RealESRGAN_x4plus -i ../thesis/test_AID_x2 --outscale 4