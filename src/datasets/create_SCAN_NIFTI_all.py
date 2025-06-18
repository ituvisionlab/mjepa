# ---------------------------------------------------
# STEP 1: CREATE SCAN_NIFTI_all.csv and only_hippo.csv
#          Using visit date-specific metadata (VISITYR, VISITMO, VISITDAY)
# ---------------------------------------------------

import os
import pandas as pd
import re

# Paths
DATA_ROOT = "/gpfs/data/sodicksonlab/gozde/SCAN/SCAN_NIFTI"
META_PATH = "/gpfs/data/sodicksonlab/gozde/SCAN/investigator_nacc69.csv"
ALL_CSV = "SCAN_NIFTI_all.csv"
HIPPO_CSV = "SCAN_NIFTI_only_hippo.csv"

# Load metadata with relevant columns
use_cols = ["NACCID", "ACU", "SEX", "BIRTHYR", "WEIGHT", "VISITYR", "VISITMO", "VISITDAY"]
meta_df = pd.read_csv(META_PATH, usecols=use_cols, low_memory=False)
meta_df['NACCID'] = meta_df['NACCID'].astype(str).str.zfill(9)

# Combine visit date into single column for date matching
meta_df['VISITDATE'] = pd.to_datetime(dict(year=meta_df['VISITYR'], month=meta_df['VISITMO'], day=meta_df['VISITDAY']), errors='coerce')

# Prepare for subject-visit matching
rows = []
for subject_id in os.listdir(DATA_ROOT):
    subject_path = os.path.join(DATA_ROOT, subject_id)
    if not os.path.isdir(subject_path):
        continue

    subj_meta = meta_df[meta_df['NACCID'] == subject_id]
    if subj_meta.empty:
        continue

    for fname in os.listdir(subject_path):
        if not fname.endswith(".nii.gz"):
            continue

        fpath = os.path.join(subject_path, fname)

        # Extract scan date from filename
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
        if not date_match:
            continue
        date_acquired = date_match.group(1)
        scan_date = pd.to_datetime(date_acquired)

        # Find the closest metadata row matching this scan date
        visit_meta = subj_meta.copy()
        visit_meta['delta'] = (visit_meta['VISITDATE'] - scan_date).abs()
        visit_meta = visit_meta.sort_values('delta')

        if visit_meta.empty or pd.isna(visit_meta.iloc[0]['ACU']):
            continue

        row_meta = visit_meta.iloc[0]

        label = int(row_meta['ACU']) if not pd.isna(row_meta['ACU']) else None
        if label not in [1, 2, 3, 4]:
            continue

        birth_year = row_meta['BIRTHYR']
        subject_age = 2025 - int(birth_year) if not pd.isna(birth_year) else "N/A"

        sex = row_meta['SEX']
        subject_sex = "M" if sex == 1 else "F" if sex == 2 else "N/A"

        weight = row_meta['WEIGHT']
        subject_weight = round(float(weight) * 0.453592, 2) if not pd.isna(weight) else "N/A"

        contrast_parts = fname.split("__")[0].split("_")
        contrast = contrast_parts[-1] if contrast_parts else "N/A"

        rows.append([
            label, subject_id, contrast, date_acquired,
            subject_sex, subject_age, subject_weight, fpath,
            -1, -1, -1, -1, -1, -1
        ])

columns = ["label", "subject_id", "contrast", "date_acquired", "subject_sex", "subject_age",
           "subject_weight", "nii_file_path", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
df_all = pd.DataFrame(rows, columns=columns)
df_all.to_csv(ALL_CSV, index=False)
df_all[df_all["nii_file_path"].str.contains("Hippo", case=False)].to_csv(HIPPO_CSV, index=False)
