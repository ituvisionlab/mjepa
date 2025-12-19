#!/bin/bash
module load anaconda3
source /gpfs/home/unalg01/miniconda3/etc/profile.d/conda.sh
conda activate gozdessl

RUNDIR=/gpfs/home/unalg01/jepa
cd $RUNDIR

python -m evals.main_distributed \
  --fname configs/evals/vitb16_mri_eval_mae2.3.k64_mci.yaml \
  --time 4200 \
  --nodes 1 \
  --folder /gpfs/home/unalg01/jepa \
  --partition radiology
