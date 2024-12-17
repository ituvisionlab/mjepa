#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:2
#SBATCH --mem=192GB
#SBATCH --job-name=sslgoz
#SBATCH --mail-type=END
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/slurmDistrib_%j.log
module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl
RUNDIR=/gpfs/home/unalg01/jepa
cd $RUNDIR
python -m evals.main_distributed --fname configs/evals/vitb16_mri_eval_aug.yaml --folder /gpfs/home/unalg01/jepa --partition a100_short
