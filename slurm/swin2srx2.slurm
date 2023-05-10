#!/bin/bash
#SBATCH --job-name=swinir_x2
#SBATCH --output=/scratch/s2630575/logs/slurm/%x_%A.out
#SBATCH --partition=graceTST

python -u main_test_swin2sr.py> /scratch/s2630575/logs/python_output/swin2sr_x2.txt --task classical_sr --scale 2 --training_patch_size 64 --model_path model_zoo/swin2sr/Swin2SR_ClassicalSR_X2_64.pth --folder_lq ../thesis/test_AID_x2 --folder_gt ../thesis/test_AID