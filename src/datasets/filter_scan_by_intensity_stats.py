import pandas as pd
import nibabel as nib
import numpy as np
import os

# Input CSV and output paths
INPUT_CSV = "SCAN_NIFTI_all_final.csv"
OUTPUT_CSV = "SCAN_NIFTI_filtered_by_stats.csv"

# Thresholds to identify suspect volumes (tune as needed)
MAX_ALLOWED_MEAN = 2000.0    # Map volumes often have very high mean
MIN_ALLOWED_STD = 10.0       # Map volumes often have very low variance

# Load the CSV
df = pd.read_csv(INPUT_CSV)

# Store retained rows and skipped ones for review
retained_rows = []
skipped_files = []

for i, row in df.iterrows():
    path = row["nii_file_path"]
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        continue

    try:
        data = nib.load(path).get_fdata()
        data = data[np.isfinite(data)]  # Remove any NaNs or infs

        mean_val = data.mean()
        std_val = data.std()

        if mean_val > MAX_ALLOWED_MEAN or std_val < MIN_ALLOWED_STD:
            print(f"[SKIP] {os.path.basename(path)} — mean={mean_val:.1f}, std={std_val:.1f}")
            skipped_files.append((path, mean_val, std_val))
            continue

        retained_rows.append(row)

    except Exception as e:
        print(f"[ERROR] Failed to process {path}: {e}")
        continue

# Save filtered CSV
filtered_df = pd.DataFrame(retained_rows)
filtered_df.to_csv(OUTPUT_CSV, index=False)
print(f"[DONE] Filtered dataset saved to {OUTPUT_CSV}")
print(f"[INFO] Skipped {len(skipped_files)} files based on mean/std thresholds.")
