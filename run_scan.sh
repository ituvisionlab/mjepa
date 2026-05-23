#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64GB
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

#python /gpfs/home/unalg01/jepa/src/datasets/generate_scan_all_updated.py --nodes 1 --partition gpu4_medium
#python generate_scan_no_metadata_for_pretraining.py --nodes 1 --partition gpu4_medium
#python validate_scan_labels.py --nodes 1 --partition gpu4_medium
python validate_scan_all_bet_nii.py --nodes 1 --partition gpu4_medium