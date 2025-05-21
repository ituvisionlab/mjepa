import os
import pandas as pd
from tqdm import tqdm

# ---- CONFIG ----
nifti_root = "/gpfs/data/prostatelab/NIFTI"
metadata_file = "/gpfs/data/prostatelab/NIFTI_csv/Prostate_training_6May2025.csv"
output_csv = "/gpfs/home/unalg01/jepa/src/datasets/Prostate_pretraining.csv"

# ---- LOAD METADATA ----
metadata = pd.read_csv(metadata_file, dtype=str)
print(f"Loaded metadata: {len(metadata)} rows")

existing_subjects = set(os.listdir(nifti_root))
print(f"Found {len(existing_subjects)} folders in NIFTI root")

rows = []

# ---- MAIN LOOP ----
for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
    acc_num = row["AccNum"]
    patient_id = row["PatientID"]
    series_date = row["SeriesDate"]
    age = row.get("Age", "")
    max_pirads = row.get("MaxPIRADS", "")

    if acc_num not in existing_subjects:
        continue  # Skip if NIFTI folder is missing

    subject_dir = os.path.join(nifti_root, acc_num)
    t2_file = os.path.join(subject_dir, "axt2.nii.gz")
    adc_file = os.path.join(subject_dir, "adc.nii.gz")

    if not (os.path.isfile(t2_file) and os.path.isfile(adc_file)):
        continue  # Skip if both contrasts are not available

    # ---- Add row with label cast to integer if possible ----
    try:
        label = int(float(max_pirads))  # handles '2.0', '3.0', etc.
    except ValueError:
        continue  # Skip rows with non-numeric PIRADS

    rows.append({
        "label": label,
        "patient_id": patient_id,
        "acc_num": acc_num,
        "series_date": series_date,
        "age": age,
        "t2_path": t2_file,
        "adc_path": adc_file
    })

# ---- SAVE TO CSV ----
df_out = pd.DataFrame(rows)
df_out.to_csv(output_csv, index=False)
print(f"Saved {len(df_out)} rows to {output_csv}")
