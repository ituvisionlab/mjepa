#!/bin/bash
#SBATCH --job-name=wandb-test
#SBATCH --output=wandb_test.out
#SBATCH --error=wandb_test.err
#SBATCH --partition=a100_short
#SBATCH --ntasks=1
#SBATCH --time=5

module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl

python test_wandb_access.py
