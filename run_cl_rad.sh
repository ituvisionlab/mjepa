#!/bin/bash
#SBATCH --partition=radiology
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128GB
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
python -m evals.main_distributed --fname configs/evals/vitb16_mri_eval2.yaml --time 4300 --nodes 1  --folder /gpfs/home/unalg01/jepa --partition radiology
