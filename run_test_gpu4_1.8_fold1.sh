#!/bin/bash
module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl

RUNDIR=/gpfs/home/unalg01/jepa
cd $RUNDIR

python -m evals.main_distributed \
  --fname configs/evals/vitb16_mri_test1.8_fold1.yaml \
  --time 300 \
  --nodes 1 \
  --folder /gpfs/home/unalg01/jepa \
  --partition gpu4_short
