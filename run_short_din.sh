#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=192GB
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=launch
#SBATCH --mail-type=END
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/slurmDistrib_%j.log
module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl
RUNDIR=/gpfs/home/unalg01/jepa
cd $RUNDIR
python -m app.main_distributed --fname configs/pretrain_dino/vitb16_mri_dino.yaml --time 4300 --nodes 1  --folder /gpfs/home/unalg01/jepa --partition a100_short
