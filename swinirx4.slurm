#!/bin/bash
#SBATCH --job-name=swinir_x4
#SBATCH --output=/scratch/s2630575/logs/slurm/%x_%A.out
#SBATCH --partition=graceTST

python -u main_test_swinir.py> /scratch/s2630575/logs/python_output/swinir_x4.txt --task classical_sr --scale 4 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth --folder_lq ../thesis/test_AID_x4 --folder_gt ../thesis/test_AID