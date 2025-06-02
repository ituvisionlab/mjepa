import os
import pandas as pd
from tqdm import tqdm

# ---- CONFIG ----
nifti_root = "/gpfs/data/prostatelab/NIFTI"
metadata_file = "/gpfs/data/prostatelab/NIFTI_csv/Prostate_training_6May2025.csv"
bx_files = [
    "/gpfs/data/prostatelab/NIFTI_csv/Bx_train_set_2025_11_4.csv",
    "/gpfs/data/prostatelab/NIFTI_csv/Bx_val_set_2025_11_4.csv",
    "/gpfs/data/prostatelab/NIFTI_csv/Bx_test_set_2025_17_1.csv"
]
output_csv_single = "/gpfs/home/unalg01/jepa/src/datasets/Prostate_pretraining_single.csv"
output_csv_multi = "/gpfs/home/unalg01/jepa/src/datasets/Prostate_pretraining_multichannel.csv"

# ---- LOAD METADATA ----
metadata = pd.read_csv(metadata_file, dtype=str)
print(f"Loaded metadata: {len(metadata)} rows")

# ---- LOAD DOWNSTREAM SUBJECTS TO EXCLUDE ----
bx_accessions = set()
for bx_file in bx_files:
    bx_df = pd.read_csv(bx_file)
    bx_accessions.update(bx_df['AccessionNumber'].astype(str))
print(f"Excluding {len(bx_accessions)} downstream subjects (from Bx files)")

# ---- MAIN LOOP ----
rows_single = []
rows_multi = []

existing_subjects = set(os.listdir(nifti_root))
print(f"Found {len(existing_subjects)} folders in NIFTI root")

for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
    acc_num = row["AccNum"]
    if acc_num in bx_accessions or acc_num not in existing_subjects:
        continue

    subject_dir = os.path.join(nifti_root, acc_num)
    t2_path = os.path.join(subject_dir, "axt2.nii.gz")
    adc_path = os.path.join(subject_dir, "adc.nii.gz")
    b1500_path = os.path.join(subject_dir, "b1500.nii.gz")

    if not (os.path.isfile(t2_path) and os.path.isfile(adc_path) and os.path.isfile(b1500_path)):
        continue  # skip if any file is missing

    patient_id = row["PatientID"]
    series_date = row["SeriesDate"]
    age = row.get("Age", "")
    max_pirads = row.get("MaxPIRADS", "")

    try:
        label = int(float(max_pirads)) - 1  # PIRADS from 1-5 to 0-4
    except ValueError:
        continue

    # --- MULTI-CHANNEL ENTRY ---
    rows_multi.append({
        "label": label,
        "patient_id": patient_id,
        "acc_num": acc_num,
        "series_date": series_date,
        "age": age,
        "adc_path": adc_path,
        "axt2_path": t2_path,
        "b1500_path": b1500_path
    })

    # --- SINGLE-CHANNEL UNIFIED FORMAT (for brain/prostate compatibility) ---
    for contrast, path in zip(['ADC', 'AXT2', 'B1500'], [adc_path, t2_path, b1500_path]):
        rows_single.append({
            "label": label,
            "subject_id": patient_id,
            "contrast": contrast,
            "date_acquired": series_date,
            "subject_age": age,
            "nii_file_path": path
        })

# ---- SAVE TO CSV ----
pd.DataFrame(rows_multi).to_csv(output_csv_multi, index=False)
pd.DataFrame(rows_single).to_csv(output_csv_single, index=False)

print(f" Saved multichannel: {len(rows_multi)} rows to {output_csv_multi}")
print(f" Saved single-channel: {len(rows_single)} rows to {output_csv_single}")
