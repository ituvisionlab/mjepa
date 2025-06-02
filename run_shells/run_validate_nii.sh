#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --job-name=validate
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/download_fastmri_%j.log

# Load your Python environment or module if needed
module load python/3.8  # or conda activate <env>

# Run your Python script
python src/datasets/validate_nifti_volumes_and_move_bad.py