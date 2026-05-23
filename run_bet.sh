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
python -m generate_abide_filtered_master --time 500 --nodes 1 --partition gpu4_medium
#python -m validate_scan_all_bet_nii --time 500 --nodes 1 --partition gpu4_medium
