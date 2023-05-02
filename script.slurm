#!/bin/bash
#SBATCH --job-name=swinir_x3
#SBATCH --output=/scratch/s2630575/logs/slurm/%x_%A.out
#SBATCH --partition=graceTST

cd SwinIR

python -u main_test_swinir.py> /scratch/s2630575/logs/python_output/swinir_output.txt --task classical_sr --scale 3 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth --folder_lq ../thesis/test_AID_x3 --folder_gt ../thesis/test_AID