#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=a100_long
#SBATCH --tasks-per-node=8
#SBATCH --time=4-00:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=sslgoz
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/slurm_%j.log
module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl
RUNDIR=/gpfs/home/unalg01/jepa
cd $RUNDIR
python -m app.main --fname configs/pretrain/vitl16single.yaml

