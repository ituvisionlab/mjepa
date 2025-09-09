#!/bin/bash
module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl

RUNDIR=/gpfs/home/unalg01/jepa
cd $RUNDIR

python -m evals.main_distributed \
  --fname configs/evals/resnet3d_ad_nc_k8.yaml \
  --time 3800 \
  --nodes 1 \
  --folder /gpfs/home/unalg01/jepa \
  --partition a100_short
