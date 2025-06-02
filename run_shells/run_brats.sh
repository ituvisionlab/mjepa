#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --job-name=sslgoz
#SBATCH --mail-type=END
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/slurmDistrib_%j.log
module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl
RUNDIR=/gpfs/home/unalg01/jepa/src/datasets
cd $RUNDIR
python -m create_brats --time 5300 --nodes 1 --folder /gpfs/home/unalg01/jepa --partition gpu4_medium
