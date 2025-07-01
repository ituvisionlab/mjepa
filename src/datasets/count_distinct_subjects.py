import pandas as pd

# Replace with your actual CSV path
#csv_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_cv_folds_stratified/adni_pretrain.csv'
#csv_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_cv_folds_stratified/adni_downstream.csv'
#csv_path = '/gpfs/home/unalg01/jepa/src/datasets/oasis3_all_bet.csv'
#csv_path = '/gpfs/home/unalg01/jepa/src/datasets/ppmi_all_bet_nii_with_bbox.csv'
#csv_path = '/gpfs/home/unalg01/jepa/src/datasets/brats24_all_nii.csv'
#csv_path = '/gpfs/home/unalg01/jepa/src/datasets/ixi_all_bet_nii_with_bbox.csv'
#csv_path = '/gpfs/home/unalg01/jepa/src/datasets/scan_pretraining.csv'
#csv_path = '/gpfs/home/unalg01/jepa/src/datasets/scan_downstream_pool.csv'
#csv_path = '/gpfs/home/unalg01/jepa/src/datasets/ucsf_all_nii.csv'
csv_path = '/gpfs/home/unalg01/jepa/src/datasets/ABIDE_master_with_betmask.csv'

# Load the CSV
df = pd.read_csv(csv_path)

# Count unique subject_ids
unique_subjects = df['subject_id'].nunique()

print(f"Number of distinct subject IDs: {unique_subjects}")

#Pretrains:
#ADNI: Number of distinct subject IDs: 1224
#OASIS3: Number of distinct subject IDs: 1376
#PPMI: Number of distinct subject IDs: 356
#BRATS24: Number of distinct subject IDs: 731
#IXI: Number of distinct subject IDs: 581
#Mood: Number of distinct subject IDs: 
#SCAN: Number of distinct subject IDs: 3701

#Downstreams:
# ADNI: Number of distinct subject IDs: 496
# SCAN: Number of distinct subject IDs: 1176
# UCSF:Number of distinct subject IDs: 495
# ABIDE: Number of distinct subject IDs: 1109