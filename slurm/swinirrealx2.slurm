#!/bin/bash
#SBATCH --job-name=swinirreal_x2
#SBATCH --output=/scratch/s2630575/logs/slurm/%x_%A.out
#SBATCH --partition=graceTST

python -u main_test_swinir.py --task real_sr --scale 2 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth --folder_lq ../thesis/test_AID