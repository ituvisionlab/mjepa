#!/bin/bash
module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl

RUNDIR=/gpfs/home/unalg01/jepa
cd $RUNDIR

python -m app.main_distributed \
  --fname configs/pretrain_mae/vitb16_mri_mae.yaml \
  --time 4700 \
  --nodes 1 \
  --folder /gpfs/home/unalg01/jepa \
  --partition radiology
