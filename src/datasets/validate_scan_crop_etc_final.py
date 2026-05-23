import os
import nibabel as nib
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

# --- Config ---
CSV_PATH = "/gpfs/home/unalg01/jepa/src/datasets/SCAN_NIFTI_all_with_cleaned_contrast.csv"
LOG_FILE = "/gpfs/home/unalg01/jepa/src/datasets/scan_log_bad_nifti_files.txt"
DELTA_BOX = 6
BBOX_FIELDS = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]

# --- Crop helper (as in the training logic) ---
def crop_volume_bbox(volume, bbox, delta_box=6):
    if -1 in bbox or len(bbox) != 6:
        return volume
    x1, x2, y1, y2, z1, z2 = bbox
    x1, x2 = max(0, x1 - delta_box), min(volume.shape[0], x2 + delta_box)
    y1, y2 = max(0, y1 - delta_box), min(volume.shape[1], y2 + delta_box)
    z1, z2 = max(0, z1 - delta_box), min(volume.shape[2], z2 + delta_box)
    return volume[x1:x2, y1:y2, z1:z2]

# --- Load CSV ---
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} entries from CSV")

bad_entries = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    path = row["nii_file_path"]
    bbox = [row.get(f, -1) for f in BBOX_FIELDS]

    if not os.path.exists(path):
        bad_entries.append((path, "MISSING"))
        continue

    try:
        img = nib.load(path)
        volume = img.get_fdata()

        if 0 in volume.shape:
            bad_entries.append((path, f"ZERO_SHAPE: {volume.shape}"))
            continue

        vol_cropped = crop_volume_bbox(volume, bbox, DELTA_BOX)
        if 0 in vol_cropped.shape:
            bad_entries.append((path, f"EMPTY_AFTER_CROP: {vol_cropped.shape}"))

    except Exception as e:
        bad_entries.append((path, f"LOAD_ERROR: {e}"))

# --- Write results ---
with open(LOG_FILE, 'w') as f:
    for path, reason in bad_entries:
        f.write(f"{path}\t{reason}\n")

print(f"[DONE] Checked {len(df)} files. Found {len(bad_entries)} problematic entries.")
print(f"[LOG] Results written to {LOG_FILE}")
