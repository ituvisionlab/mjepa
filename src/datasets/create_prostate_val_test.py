import os
import pandas as pd

# ---- CONFIG ----
nifti_root = "/gpfs/data/prostatelab/NIFTI"
pretrain_file = "/gpfs/home/unalg01/jepa/src/datasets/Prostate_pretraining_multichannel.csv"
log_file = "/gpfs/home/unalg01/jepa/src/datasets/overlapping_pretrain_subjects.log"

# Downstream sources
downstream_files = {
    "train": ["/gpfs/data/prostatelab/NIFTI_csv/Bx_train_set_2025_11_4.csv"],
    "val": [
        "/gpfs/data/prostatelab/NIFTI_csv/Prostate_validation_6May2025.csv",
        "/gpfs/data/prostatelab/NIFTI_csv/Bx_val_set_2025_11_4.csv"
    ],
    "test": [
        "/gpfs/data/prostatelab/NIFTI_csv/Prostate_test_6May2025.csv",
        "/gpfs/data/prostatelab/NIFTI_csv/Bx_test_set_2025_17_1.csv"
    ]
}

# Output CSVs
out_csvs = {
    "train": "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_train.csv",
    "val": "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_val.csv",
    "test": "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_test.csv"
}

# ---- LOAD PRETRAIN SUBJECTS ----
pretrain_df = pd.read_csv(pretrain_file)
pretrain_subjects = set(pretrain_df["acc_num"].astype(str))
print(f"Loaded {len(pretrain_subjects)} subjects from pretraining set.")

# ---- EXISTING SUBJECTS IN NIFTI ROOT ----
available_subjects = set(s.strip().split(".")[0] for s in os.listdir(nifti_root) if os.path.isdir(os.path.join(nifti_root, s)))
print(f"Detected {len(available_subjects)} available subject folders in {nifti_root}")

# ---- HELPER FUNCTION ----
def load_and_filter_metadata(metadata_file, available_subjects, pretrain_subjects, overlaps_log):
    df = pd.read_csv(metadata_file, dtype=str)
    rows = []

    subject_col = 'AccessionNumber' if 'AccessionNumber' in df.columns else 'AccNum'
    series_date_col = 'StudyDate' if 'StudyDate' in df.columns else 'SeriesDate'

    for _, row in df.iterrows():
        subject_id = row.get(subject_col)
        if subject_id is None:
            continue

        # Normalize: remove decimals like 39021867.0 → 39021867 and strip whitespace
        subject_id = str(subject_id).strip().split(".")[0]


        if subject_id in pretrain_subjects:
            overlaps_log.add(subject_id)
            continue
        if subject_id not in available_subjects:
            continue

        series_date = row.get(series_date_col, "")
        age = row.get("Age", "")
        max_pirads = row.get("MaxPIRADS") or row.get("maxPIRADS") or ""

        subject_dir = os.path.join(nifti_root, subject_id)
        t2_file = os.path.join(subject_dir, "axt2.nii.gz")
        adc_file = os.path.join(subject_dir, "adc.nii.gz")
        b1500_file = os.path.join(subject_dir, "b1500.nii.gz")

        try:
            label = int(float(max_pirads)) - 1
        except ValueError:
            print(f"[SKIPPED] {subject_id}: invalid PIRADS value → {max_pirads}")
            continue

        # Optional fields with safe fallbacks
        def get_numeric(row, col):
            try:
                return float(row.get(col, -1))
            except:
                return -1

        gleason_score = get_numeric(row, "MaxGleasonScore")
        psa = get_numeric(row, "parsed_psa")
        volume = get_numeric(row, "parsed_prostate_volume")

        rows.append({
            "label": label,
            "subject_id": subject_id,
            "acc_num": subject_id,
            "series_date": series_date,
            "age": age,
            "axt2_path": t2_file,
            "adc_path": adc_file,
            "b1500_path": b1500_file,
            "psa": psa,
            "gleason_score": gleason_score,
            "prostate_volume": volume
        })

    return pd.DataFrame(rows)

# ---- PROCESS EACH SPLIT ----
overlapping_subjects = set()

for split in ["train", "val", "test"]:
    files = downstream_files[split]
    combined_df = pd.concat(
        [load_and_filter_metadata(f, available_subjects, pretrain_subjects, overlapping_subjects) for f in files],
        ignore_index=True
    )
    combined_df.to_csv(out_csvs[split], index=False)
    print(f" Saved {split} split with {len(combined_df)} subjects to {out_csvs[split]}")

# ---- SAVE OVERLAP LOG ----
with open(log_file, 'w') as f:
    for acc in sorted(overlapping_subjects):
        f.write(f"{acc}\n")

print(f" Excluded {len(overlapping_subjects)} overlapping subjects")
print(f" Logged overlaps to {log_file}")
