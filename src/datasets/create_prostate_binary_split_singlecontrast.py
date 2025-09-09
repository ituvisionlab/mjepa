import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---- CONFIG ----
nifti_root = "/gpfs/data/prostatelab/NIFTI"
metadata_file = "/gpfs/data/prostatelab/NIFTI_csv/Prostate_training_6May2025.csv"
pretrain_csv_single = "/gpfs/home/unalg01/jepa/src/datasets/Prostate_pretraining_single.csv"

output_dir = "/gpfs/home/unalg01/jepa/src/datasets/"
os.makedirs(output_dir, exist_ok=True)

# Output CSVs for downstream
output_csvs = {
    "train": os.path.join(output_dir, "Prostate_downstream_train_single.csv"),
    "val": os.path.join(output_dir, "Prostate_downstream_val_single.csv"),
    "test": os.path.join(output_dir, "Prostate_downstream_test_single.csv"),
}

# ---- LOAD METADATA ----
metadata = pd.read_csv(metadata_file, dtype=str)
pretrain_df = pd.read_csv(pretrain_csv_single, dtype=str)

# Subjects used in pretraining
pretrain_subjects = set(pretrain_df['subject_id'].unique())
print(f"Excluding {len(pretrain_subjects)} pretraining subjects")

# ---- FILTER UNUSED SAMPLES FOR DOWNSTREAM ----
downstream_rows = []

for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
    patient_id = row["PatientID"]
    if patient_id in pretrain_subjects:
        continue

    acc_num = row["AccNum"]
    subject_dir = os.path.join(nifti_root, acc_num)
    series_date = row["SeriesDate"]
    age = row.get("Age", "")
    max_pirads = row.get("MaxPIRADS", "")

    try:
        pirads_label = int(float(max_pirads))
    except ValueError:
        continue  # Skip if label invalid

    if pirads_label in [1, 2, 3]:
        binary_label = 0
    elif pirads_label in [4, 5]:
        binary_label = 1
    else:
        continue  # Unexpected labels skipped

    contrast_files = {
        'ADC': os.path.join(subject_dir, "adc.nii.gz"),
        'AXT2': os.path.join(subject_dir, "axt2.nii.gz"),
        'B1500': os.path.join(subject_dir, "b1500.nii.gz")
    }

    for contrast, path in contrast_files.items():
        if os.path.isfile(path):
            downstream_rows.append({
                "label": binary_label,
                "subject_id": patient_id,
                "contrast": contrast,
                "date_acquired": series_date,
                "subject_age": age,
                "nii_file_path": path
            })

print(f"Total downstream rows collected: {len(downstream_rows)}")
df_downstream = pd.DataFrame(downstream_rows)

# ---- STRATIFIED SPLIT INTO TRAIN/VAL/TEST ----
df_train, df_temp = train_test_split(df_downstream, test_size=0.3, random_state=42, stratify=df_downstream["label"])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp["label"])

splits = {'train': df_train, 'val': df_val, 'test': df_test}

# ---- SAVE SPLITS TO CSV ----
for split_name, split_df in splits.items():
    class_counts = split_df["label"].value_counts().to_dict()
    print(f"\n Split: {split_name}")
    print(f" Total samples: {len(split_df)}")
    print(f"   ➤ Class 0 (PIRADS 1–3): {class_counts.get(0, 0)}")
    print(f"   ➤ Class 1 (PIRADS 4–5): {class_counts.get(1, 0)}")

    split_df.to_csv(output_csvs[split_name], index=False)
    print(f" Saved split '{split_name}' to {output_csvs[split_name]}")
