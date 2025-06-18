import os
import re
import ast
import pandas as pd
import numpy as np
import nibabel as nib
from datetime import datetime

# --- Paths ---
DATA_ROOT = "/gpfs/data/sodicksonlab/gozde/SCAN/SCAN_NIFTI"
LOG_FILE = "scan_dataset_summary.log"
OUTPUT_CSV = "SCAN_NIFTI_pretraining_only.csv"
LOG_OUT = "scan_pretraining_log.txt"

# --- Filtering Criteria ---
VALID_CONTRASTS = ["T1", "T2", "MPRAGE", "FLAIR", "_IR_"]
MIN_FILE_SIZE = 1 * 1024 * 1024
MIN_FOV = 50
MAX_SPACING = 6.5

# --- Parse subject list from log ---
def extract_skipped_ids_from_log(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()

    inside_block = False
    list_lines = []

    for line in lines:
        if "List of subjects with missing metadata:" in line:
            inside_block = True
        elif inside_block:
            list_lines.append(line.strip())
            if "]" in line:
                break

    full_text = " ".join(list_lines)
    try:
        ids = ast.literal_eval(full_text)
        return [sid for sid in ids if isinstance(sid, str) and sid.startswith("NACC")]
    except Exception as e:
        print(f"[ERROR] Could not parse skipped subject list: {e}")
        print(f"[DEBUG] Full extracted block:\n{full_text}")
        raise

# --- Filters ---
def calculate_bbox(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        if data.ndim != 3 or not np.any(data):
            return None
        coords = np.argwhere(data > 0)
        return {
            "xmin": coords[:, 0].min(), "xmax": coords[:, 0].max(),
            "ymin": coords[:, 1].min(), "ymax": coords[:, 1].max(),
            "zmin": coords[:, 2].min(), "zmax": coords[:, 2].max()
        }
    except Exception as e:
        print(f"[ERROR] bbox for {file_path}: {e}")
        return None

def passes_filters(fpath):
    try:
        if os.path.getsize(fpath) < MIN_FILE_SIZE:
            return False
        img = nib.load(fpath)
        data = img.get_fdata()
        if data.ndim != 3:
            return False
        spacing = img.header.get_zooms()[:3]
        shape = data.shape[:3]
        fov = [s * d for s, d in zip(spacing, shape)]
        return all(f >= MIN_FOV for f in fov) and all(s <= MAX_SPACING for s in spacing)
    except:
        return False

# --- Run ---
skipped_subjects = extract_skipped_ids_from_log(LOG_FILE)

rows = []
added_subjects = set()
skipped_subjects_final = {}

for subject_id in skipped_subjects:
    subj_path = os.path.join(DATA_ROOT, subject_id)
    if not os.path.isdir(subj_path):
        skipped_subjects_final[subject_id] = "Folder not found"
        continue

    any_valid = False
    for fname in os.listdir(subj_path):
        if not fname.endswith(".nii.gz") or "_betmask" in fname:
            continue
        if not any(k.lower() in fname.lower() for k in VALID_CONTRASTS):
            continue

        fpath = os.path.join(subj_path, fname)
        if not passes_filters(fpath):
            continue

        base = fname.replace(".nii.gz", "")
        betmask_path = os.path.join(subj_path, f"{base}_betmask.nii.gz")
        betmask_mask_path = os.path.join(subj_path, f"{base}_betmask_mask.nii.gz")

        if not os.path.exists(betmask_path) or not os.path.exists(betmask_mask_path):
            continue

        bbox = calculate_bbox(betmask_mask_path)
        if not bbox:
            continue

        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
        date_acquired = date_match.group(1) if date_match else "Unknown"

        contrast_parts = fname.split("__")[0].split("_")
        contrast = contrast_parts[-1] if contrast_parts else "N/A"

        rows.append({
            "label": -1,
            "subject_id": subject_id,
            "contrast": contrast,
            "date_acquired": date_acquired,
            "subject_sex": "N/A",
            "subject_age": "N/A",
            "subject_weight": "N/A",
            "nii_file_path": betmask_path,
            **bbox
        })
        added_subjects.add(subject_id)
        any_valid = True

    if not any_valid:
        skipped_subjects_final[subject_id] = "No valid NIfTI file after filters"

# --- Save CSV ---
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[DONE] Pretraining-only CSV written to {OUTPUT_CSV} with {len(rows)} entries.")

# --- Write log ---
with open(LOG_OUT, "w") as logf:
    logf.write(f"Pretraining CSV created at {datetime.now()}\n")
    logf.write(f"Total subjects parsed from log: {len(skipped_subjects)}\n")
    logf.write(f"Subjects with valid entries added: {len(added_subjects)}\n")
    logf.write(f"Subjects skipped entirely: {len(skipped_subjects_final)}\n\n")
    for sid, reason in skipped_subjects_final.items():
        logf.write(f"{sid}: {reason}\n")

print(f"[LOG] Detailed summary written to {LOG_OUT}")
