import pandas as pd
import nibabel as nib
import numpy as np
import os

# Input CSV
CSV_PATH = "SCAN_NIFTI_all_final.csv"
LOG_PATH = "volume_intensity_stats.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

# List to hold stats for each volume
stats = []

for i, row in df.iterrows():
    path = row["nii_file_path"]
    subject_id = row.get("subject_id", "N/A")
    contrast = row.get("contrast", "N/A")

    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        continue

    try:
        data = nib.load(path).get_fdata()
        data = data[np.isfinite(data)]  # Remove NaNs or infs

        mean_val = data.mean()
        std_val = data.std()

        stats.append({
            "subject_id": subject_id,
            "contrast": contrast,
            "filename": os.path.basename(path),
            "mean_intensity": round(mean_val, 2),
            "std_intensity": round(std_val, 2)
        })

    except Exception as e:
        print(f"[ERROR] Failed on {path}: {e}")
        continue

# Save stats to CSV
stats_df = pd.DataFrame(stats)
stats_df.to_csv(LOG_PATH, index=False)

print(f"[DONE] Logged intensity stats to: {LOG_PATH}")
