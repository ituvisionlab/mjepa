import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ---- CONFIG ----
nifti_root = "/gpfs/data/prostatelab/NIFTI"
train_metadata_file = "/gpfs/data/prostatelab/NIFTI_csv/Prostate_training_6May2025.csv"
val_metadata_file = "/gpfs/data/prostatelab/NIFTI_csv/Prostate_validation_6May2025.csv"
test_metadata_file = "/gpfs/data/prostatelab/NIFTI_csv/Prostate_test_6May2025.csv"
valid_subjects_file = "/gpfs/data/prostatelab/NIFTI_csv/valid_subjects.csv"

out_train_csv = "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_train.csv"
out_val_csv = "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_val.csv"
out_test_csv = "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_test.csv"

# ---- LOAD VALID SUBJECTS ----
valid_subjects_df = pd.read_csv(valid_subjects_file, dtype=str)
valid_acc_nums = set(valid_subjects_df["subject_id"].astype(str))

def load_and_filter_metadata(metadata_file, valid_acc_nums):
    df = pd.read_csv(metadata_file, dtype=str)
    rows = []

    for _, row in df.iterrows():
        acc_num = row["AccNum"]
        if acc_num not in valid_acc_nums:
            continue

        patient_id = row["PatientID"]
        series_date = row["SeriesDate"]
        age = row.get("Age", "")
        max_pirads = row.get("MaxPIRADS", "")

        subject_dir = os.path.join(nifti_root, acc_num)
        t2_file = os.path.join(subject_dir, "axt2.nii.gz")
        adc_file = os.path.join(subject_dir, "adc.nii.gz")

        if not (os.path.isfile(t2_file) and os.path.isfile(adc_file)):
            continue

        try:
            label = int(float(max_pirads))  # handles '2.0', '3.0', etc.
        except ValueError:
            continue

        rows.append({
            "label": label,
            "patient_id": patient_id,
            "acc_num": acc_num,
            "series_date": series_date,
            "age": age,
            "t2_path": t2_file,
            "adc_path": adc_file
        })

    return pd.DataFrame(rows)

# ---- BUILD DOWNSTREAM VAL AND TEST SETS ----
val_df = load_and_filter_metadata(val_metadata_file, valid_acc_nums)
test_df = load_and_filter_metadata(test_metadata_file, valid_acc_nums)
print(f"✅ Downstream val: {len(val_df)} rows")
print(f"✅ Downstream test: {len(test_df)} rows")

# ---- BUILD DOWNSTREAM TRAIN SET ----
full_train_df = load_and_filter_metadata(train_metadata_file, valid_acc_nums)

# Stratified sample ~4000 entries
if len(full_train_df) > 4000:
    full_train_df = full_train_df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=1.0, random_state=42)
    ).sample(n=4000, random_state=42)

print(f"✅ Downstream train: {len(full_train_df)} rows")

# ---- SAVE TO CSV ----
full_train_df.to_csv(out_train_csv, index=False)
val_df.to_csv(out_val_csv, index=False)
test_df.to_csv(out_test_csv, index=False)

print("✅ CSVs saved to:")
print(f"  Train → {out_train_csv}")
print(f"  Val   → {out_val_csv}")
print(f"  Test  → {out_test_csv}")
