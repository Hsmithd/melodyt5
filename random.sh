#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --job-name="random"
#SBATCH --output=random.txt
#SBATCH --error=random.err
#SBATCH --mail-user=SMITHHD4541@UWEC.EDU
#SBATCH --mail-type=ALL

python random_model.py