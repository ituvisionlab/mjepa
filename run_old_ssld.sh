#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=a100_long
#SBATCH --tasks-per-node=2
#SBATCH --time=5-00:00:00
#SBATCH --job-name=sslgoz
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/slurmDistrib_%j.log
module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl
RUNDIR=/gpfs/home/unalg01/jepa
cd $RUNDIR
python -m app.main_distributed --fname configs/pretrain/vitl16.yaml --folder /gpfs/home/unalg01/jepa --partition a100_long
