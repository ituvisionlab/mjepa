# build_multicontrast_csv_adni.py

import pandas as pd
import os

# === INPUT / OUTPUT FILES ===
INPUT_CSV = "/gpfs/home/unalg01/jepa/src/datasets/adni_cv_folds_stratified/adni_downstream.csv"
OUTPUT_CSV = INPUT_CSV.replace(".csv", "_multicontrast.csv")
LOG_FILE = INPUT_CSV.replace(".csv", "_multicontrast.log")

# === SETTINGS ===
MAX_CONTRASTS = 6

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV)

# === PREPARE FIELDS ===
path_cols = [f"contrast{i+1}_path" for i in range(MAX_CONTRASTS)]
bbox_cols = [
    coord
    for i in range(MAX_CONTRASTS)
    for coord in [
        f"contrast{i+1}_xmin",
        f"contrast{i+1}_xmax",
        f"contrast{i+1}_ymin",
        f"contrast{i+1}_ymax",
        f"contrast{i+1}_zmin",
        f"contrast{i+1}_zmax",
    ]
]

# === GROUP BY SUBJECT & DATE ===
multi_rows = []
log_lines = []

for (subject_id, date), group in df.groupby(["subject_id", "date_acquired"]):
    label = group["label"].iloc[0]
    sex = group["subject_sex"].iloc[0]
    age = group["subject_age"].iloc[0]
    weight = group["subject_weight"].iloc[0]

    selected_scans = group.sample(min(MAX_CONTRASTS, len(group)), random_state=42)

    row = {
        "label": label,
        "subject_id": subject_id,
        "date_acquired": date,
        "subject_sex": sex,
        "subject_age": age,
        "subject_weight": weight,
    }

    # Add paths
    paths = selected_scans["nii_file_path"].tolist()
    for i, path in enumerate(paths):
        row[path_cols[i]] = path

    # Add bbox coordinates
    bbox_data = selected_scans[["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]].values
    bbox_flat = bbox_data.flatten()
    for col_name, bbox_val in zip(bbox_cols, bbox_flat):
        row[col_name] = bbox_val

    multi_rows.append(row)
    log_lines.append(f"{subject_id} on {date}: {len(paths)} contrasts")

# === SAVE OUTPUT ===
out_df = pd.DataFrame(multi_rows, columns=[
    "label", "subject_id", "date_acquired", "subject_sex", "subject_age", "subject_weight"]
    + path_cols + bbox_cols
)
out_df.to_csv(OUTPUT_CSV, index=False)

with open(LOG_FILE, "w") as f:
    f.write("ADNI multi-contrast groupings (up to 6 per row):\n")
    f.write("\n".join(log_lines))

print(f"Wrote {len(out_df)} rows to {OUTPUT_CSV} and log to {LOG_FILE}")
