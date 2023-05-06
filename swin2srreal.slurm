#!/bin/bash
#SBATCH --job-name=swin2srreal
#SBATCH --output=/scratch/s2630575/logs/slurm/swin2sr.out
#SBATCH --partition=graceTST

python main_test_swin2sr.py --task real_sr --scale 4 --model_path model_zoo/swin2sr/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth --folder_lq ../thesis/test_AID