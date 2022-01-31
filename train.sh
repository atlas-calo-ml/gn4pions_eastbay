#!/bin/bash 
#SBATCH -C gpu 
#SBATCH -A m3246_g
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-task 1
#SBATCH --time=1:00:00
#SBATCH -J test
#SBATCH -o logs/%x-%j.out

python train.py 
