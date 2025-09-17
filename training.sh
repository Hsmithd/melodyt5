#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --job-name="Mus_Train"
#SBATCH --output=Mus_Train.txt
#SBATCH --error=Mus_Train.err
#SBATCH --mail-user=SMITHHD4541@UWEC.EDU
#SBATCH --mail-type=ALL

torchrun --nproc_per_node=1 --standalone --nnodes=1 train.py
