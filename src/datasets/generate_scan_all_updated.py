import os
import re
import pandas as pd
import numpy as np
import nibabel as nib
import subprocess
from datetime import datetime

# --- Paths ---
DATA_ROOT = "/gpfs/data/sodicksonlab/gozde/SCAN/SCAN_NIFTI"
META_PATH = "/gpfs/data/sodicksonlab/gozde/SCAN/investigator_nacc69.csv"
MASTER_CSV = "SCAN_NIFTI_all_with_betmask_and_bbox.csv"
LOG_FILE = "scan_dataset_summary.log"

# --- Filtering Criteria ---
VALID_CONTRASTS = ["T1", "T2", "MPRAGE", "FLAIR", "_IR_"]
MIN_FILE_SIZE = 1 * 1024 * 1024  # 1MB
MIN_FOV = 50  # mm
MAX_SPACING = 6.5  # mm

# --- Load Metadata ---
meta_cols = ["NACCID", "NACCUDSD", "SEX", "BIRTHYR", "WEIGHT", "VISITYR", "VISITMO", "VISITDAY"]
meta_df = pd.read_csv(META_PATH, usecols=meta_cols, low_memory=False)
meta_df["NACCID"] = meta_df["NACCID"].astype(str).str.zfill(9)
meta_df["VISITDATE"] = pd.to_datetime(dict(year=meta_df["VISITYR"], month=meta_df["VISITMO"], day=meta_df["VISITDAY"]), errors="coerce")

# --- Load existing CSV ---
if os.path.exists(MASTER_CSV):
    existing_df = pd.read_csv(MASTER_CSV)
else:
    existing_df = pd.DataFrame()

# --- Utilities ---
def run_bet(input_file, output_file):
    try:
        subprocess.run(["bet", input_file, output_file, "-m"], check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] BET failed: {input_file}")

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
        print(f"[ERROR] BBox for {mask_file}: {e}")
        return None

def passes_fov_spacing_filter(img, file_path):
    try:
        data = img.get_fdata()
        if data.ndim != 3:
            return False
        header = img.header
        spacing = header.get_zooms()[:3]
        shape = header.get_data_shape()[:3]
        fov = [s * d for s, d in zip(spacing, shape)]
        return all(f >= MIN_FOV for f in fov) and all(s <= MAX_SPACING for s in spacing)
    except Exception as e:
        print(f"[ERROR] Spacing/FOV check failed for {file_path}: {e}")
        return False

# --- Initialize Tracking ---
new_rows = []
included_subjects = set()
skipped_existing_betmask = []
skipped_no_metadata = []
skipped_all_invalid_files = []

log = {
    "subjects_processed": 0,
    "files_skipped_small": 0,
    "files_skipped_fov": 0,
    "files_skipped_contrast": 0,
    "files_skipped_invalid": 0,
    "files_added": 0
}

# --- Process Each Subject Folder ---
for subject_id in os.listdir(DATA_ROOT):
    subj_path = os.path.join(DATA_ROOT, subject_id)
    if not os.path.isdir(subj_path):
        continue

    # Skip if any *_betmask.nii.gz exists
    if any(f.endswith("_betmask.nii.gz") for f in os.listdir(subj_path)):
        skipped_existing_betmask.append(subject_id)
        continue

    subj_meta = meta_df[meta_df["NACCID"] == subject_id]
    if subj_meta.empty:
        skipped_no_metadata.append(subject_id)
        continue

    any_file_added = False
    all_skipped = True

    for fname in os.listdir(subj_path):
        if not fname.endswith(".nii.gz") or "_betmask" in fname:
            continue

        if not any(k.lower() in fname.lower() for k in VALID_CONTRASTS):
            log["files_skipped_contrast"] += 1
            continue

        fpath = os.path.join(subj_path, fname)
        if os.path.getsize(fpath) < MIN_FILE_SIZE:
            log["files_skipped_small"] += 1
            continue

        try:
            img = nib.load(fpath)
            if not passes_fov_spacing_filter(img, fpath):
                log["files_skipped_fov"] += 1
                continue
        except:
            log["files_skipped_invalid"] += 1
            continue

        # BET paths
        base = fname.replace(".nii.gz", "")
        betmask_path = os.path.join(subj_path, f"{base}_betmask.nii.gz")
        betmask_mask_path = os.path.join(subj_path, f"{base}_betmask_mask.nii.gz")

        if not (os.path.exists(betmask_path) and os.path.exists(betmask_mask_path)):
            print(f"[BET] Running BET on: {fpath}")
            run_bet(fpath, betmask_path)

        if not os.path.exists(betmask_mask_path):
            continue

        bbox = calculate_bbox(betmask_mask_path)
        if not bbox:
            continue

        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
        if not date_match:
            continue
        date_acquired = date_match.group(1)
        scan_date = pd.to_datetime(date_acquired)

        visit_meta = subj_meta.copy()
        visit_meta["delta"] = (visit_meta["VISITDATE"] - scan_date).abs()
        visit_meta = visit_meta.sort_values("delta")
        row_meta = visit_meta.iloc[0] if not visit_meta.empty else None
        if row_meta is None or pd.isna(row_meta["NACCUDSD"]):
            continue

        label = int(row_meta["NACCUDSD"])
        if label not in [1, 2, 3, 4]:
            continue

        subject_sex = {1: "M", 2: "F"}.get(row_meta["SEX"], "N/A")
        subject_age = 2025 - int(row_meta["BIRTHYR"]) if not pd.isna(row_meta["BIRTHYR"]) else "N/A"
        subject_weight = round(float(row_meta["WEIGHT"]) * 0.453592, 2) if not pd.isna(row_meta["WEIGHT"]) else "N/A"

        contrast_parts = fname.split("__")[0].split("_")
        contrast = contrast_parts[-1] if contrast_parts else "N/A"

        new_rows.append({
            "label": label,
            "subject_id": subject_id,
            "contrast": contrast,
            "date_acquired": date_acquired,
            "subject_sex": subject_sex,
            "subject_age": subject_age,
            "subject_weight": subject_weight,
            "nii_file_path": betmask_path,
            **bbox
        })
        any_file_added = True
        log["files_added"] += 1
        all_skipped = False

    if any_file_added:
        included_subjects.add(subject_id)
        log["subjects_processed"] += 1
    elif all_skipped:
        skipped_all_invalid_files.append(subject_id)

# --- Save updated CSV ---
if new_rows:
    new_df = pd.DataFrame(new_rows)
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    final_df.to_csv(MASTER_CSV, index=False)
    print(f"[DONE] Added {len(new_rows)} new scans to: {MASTER_CSV}")
else:
    print("[INFO] No new valid scans found to process.")

# --- Save log summary ---
with open(LOG_FILE, "w") as f:
    f.write(f"--- SCAN Dataset Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
    f.write(f"Total new subjects processed: {log['subjects_processed']}\n")
    f.write(f"Total NIfTI files added: {log['files_added']}\n")
    f.write(f"Files skipped (small size <1MB): {log['files_skipped_small']}\n")
    f.write(f"Files skipped (FOV/spacing or non-3D): {log['files_skipped_fov']}\n")
    f.write(f"Files skipped (contrast mismatch): {log['files_skipped_contrast']}\n")
    f.write(f"Files skipped (load errors): {log['files_skipped_invalid']}\n\n")
    f.write(f"Skipped subject folders (already had betmask): {len(skipped_existing_betmask)}\n")
    f.write(f"Skipped subject folders (no metadata): {len(skipped_no_metadata)}\n")
    f.write(f"Skipped subject folders (all files invalid or filtered): {len(skipped_all_invalid_files)}\n\n")
    f.write(f"List of subjects with existing betmask files:\n{skipped_existing_betmask}\n\n")
    f.write(f"List of subjects with missing metadata:\n{skipped_no_metadata}\n\n")
    f.write(f"Subjects skipped due to all filtered files:\n{skipped_all_invalid_files}\n")

print(f"[LOG] Summary written to {LOG_FILE}")
