#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --job-name=gdown
#SBATCH --output=gdown_%j.out
#SBATCH --error=gdown_%j.err
#SBATCH --cpus-per-task=1

# Load any necessary modules or activate your environment
# module load python/3.9   # Uncomment if you use module system

# Optional: activate a virtual environment if needed
# source ~/venvs/myenv/bin/activate

# Install gdown (skip if already installed)
#pip install --user gdown

# Set your Google Drive file ID and output file name
FILE_ID="12v4o5baKmL14jMzkg1II0xq2hXuLXiI6"  # Replace with your actual file ID
OUTPUT="/gpfs/data/sodicksonlab/gozde/UCSF/UCSF-PDGM-v3.zip"         # Desired output name

# Download using gdown
gdown --id $FILE_ID -O $OUTPUT

