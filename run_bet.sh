#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=192GB
#SBATCH --nodes=1
#SBATCH --job-name=process
#SBATCH --mail-type=END
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/slurmDistrib_%j.log
module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl
module load fsl
RUNDIR=/gpfs/home/unalg01/jepa/src/datasets
cd $RUNDIR
python -m validate_adni_all_bet_nii.py --time 5300 --nodes 1 --folder /gpfs/home/unalg01/jepa --partition gpu4_medium
#python -m add_bet_masks_nii_bbox --time 5300 --nodes 1 --folder /gpfs/home/unalg01/jepa --partition gpu4_medium
