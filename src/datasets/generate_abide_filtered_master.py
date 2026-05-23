import os
import pandas as pd
import numpy as np
import nibabel as nib
import subprocess
from datetime import datetime

# === CONFIG ===
MASTER_CSV = "/gpfs/home/unalg01/jepa/src/datasets/ABIDE_master.csv"
OUTPUT_CSV = "/gpfs/home/unalg01/jepa/src/datasets/ABIDE_master_with_betmask.csv"
LOG_FILE = "/gpfs/home/unalg01/jepa/src/datasets/abide_bet_summary.log"

MIN_FILE_SIZE = 1 * 1024 * 1024  # 1 MB
MIN_FOV = 50  # mm
MAX_SPACING = 6.5  # mm

# === Load ABIDE master CSV ===
df = pd.read_csv(MASTER_CSV)

# === Utilities ===
def run_bet(input_file, output_file):
    try:
        subprocess.run(["bet", input_file, output_file, "-m"], check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"[ERROR] BET failed for {input_file}")
        return False

def calculate_bbox(mask_file):
    try:
        mask_data = nib.load(mask_file).get_fdata()
        if not np.any(mask_data):
            return None
        coords = np.argwhere(mask_data > 0)
        return {
            "xmin": coords[:, 0].min(), "xmax": coords[:, 0].max(),
            "ymin": coords[:, 1].min(), "ymax": coords[:, 1].max(),
            "zmin": coords[:, 2].min(), "zmax": coords[:, 2].max()
        }
    except Exception as e:
        print(f"[ERROR] Failed to compute bbox for {mask_file}: {e}")
        return None

def passes_filters(img):
    try:
        header = img.header
        spacing = header.get_zooms()[:3]
        shape = img.shape[:3]
        fov = [s * d for s, d in zip(spacing, shape)]
        return all(f >= MIN_FOV for f in fov) and all(s <= MAX_SPACING for s in spacing)
    except Exception as e:
        print(f"[ERROR] FOV check failed: {e}")
        return False

# === Processing ===
new_rows = []
log = {
    "total": 0,
    "skipped_small": 0,
    "skipped_invalid": 0,
    "skipped_fov": 0,
    "bet_fail": 0,
    "bbox_fail": 0,
    "processed": 0
}

for _, row in df.iterrows():
    log["total"] += 1
    nii_path = row['nii_file_path']

    if not os.path.exists(nii_path) or os.path.getsize(nii_path) < MIN_FILE_SIZE:
        log["skipped_small"] += 1
        continue

    try:
        img = nib.load(nii_path)
    except Exception:
        log["skipped_invalid"] += 1
        continue

    if not passes_filters(img):
        log["skipped_fov"] += 1
        continue

    base_path = nii_path.replace('.nii.gz', '').replace('.nii', '')
    betmask_path = base_path + "_betmask.nii.gz"
    betmask_mask_path = base_path + "_betmask_mask.nii.gz"

    # Run BET if output doesn't exist
    if not os.path.exists(betmask_path) or not os.path.exists(betmask_mask_path):
        success = run_bet(nii_path, betmask_path)
        if not success:
            log["bet_fail"] += 1
            continue

    # Calculate bbox
    bbox = calculate_bbox(betmask_mask_path)
    if not bbox:
        log["bbox_fail"] += 1
        continue

    new_rows.append({
        "label": row['label'],
        "subject_id": row['subject_id'],
        "contrast": row['contrast'],
        "date_acquired": row['date_acquired'],
        "subject_sex": row['subject_sex'],
        "subject_age": row['subject_age'],
        "subject_weight": row['subject_weight'],
        "nii_file_path": betmask_path,
        **bbox
    })

    log["processed"] += 1

# === Save Output ===
df_out = pd.DataFrame(new_rows)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"[DONE] Saved {len(df_out)} valid entries to {OUTPUT_CSV}")

# === Log Summary ===
with open(LOG_FILE, "w") as f:
    f.write(f"ABIDE BET Processing Summary ({datetime.now()}):\n")
    for k, v in log.items():
        f.write(f"{k}: {v}\n")

print(f"[LOG] Written to {LOG_FILE}")
