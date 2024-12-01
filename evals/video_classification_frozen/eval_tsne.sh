python eval_tsne.py --pretrained_path '/gpfs/data/sodicksonlab/gozde/logs/mnist/jepa-latest.pth.tar' \
    --dataset_type MRIDataset --patch_size 8 \
    --crop_size 32 --in_chans 1 --num_classes 10 \
    --data_path /gpfs/home/unalg01/jepa/src/datasets/mnist3d/nii_volumes.csv 