#!/bin/bash
#SBATCH --reservation=unal_reservation
#SBATCH --partition=reservation
#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --nodes=1
#SBATCH --job-name=launch
#SBATCH --mail-type=END
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/slurmDistrib_%j.log
module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl
RUNDIR=/gpfs/home/unalg01/jepa
cd $RUNDIR
python -m evals.main_distributed --fname configs/evals/vitb16_mri_eval_16x8.yaml --time 4300 --nodes 1 --folder /gpfs/home/unalg01/jepa --partition reservation --reservation unal_reservation
