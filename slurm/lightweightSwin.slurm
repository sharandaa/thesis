#!/bin/bash
#SBATCH --job-name=lightswin
#SBATCH --output=/scratch/s2630575/logs/slurm/lightweightswin.out
#SBATCH --partition=graceTST

cd /scratch/s2630575/SwinIR
python -u main_test_swinir.py> /scratch/s2630575/logs/python_output/lightweightclass.txt --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --folder_lq /home/s2630575/thesis/test_AID
#python -u main_test_swinir.py> /scratch/s2630575/logs/python_output/lightweightswinirx2.txt --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --folder_lq /home/s2630575/thesis/test_AID_x2 --folder_gt /home/s2630575/thesis/test_AID
#python -u main_test_swinir.py> /scratch/s2630575/logs/python_output/lightweightswinirx3.txt --task lightweight_sr --scale 3 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth --folder_lq /home/s2630575/thesis/test_AID_x3 --folder_gt /home/s2630575/thesis/test_AID
#python -u main_test_swinir.py> /scratch/s2630575/logs/python_output/lightweightswinirx4.txt --task lightweight_sr --scale 4 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth --folder_lq /home/s2630575/thesis/test_AID_x4 --folder_gt /home/s2630575/thesis/test_AID