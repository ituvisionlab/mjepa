import pandas as pd

# === INPUT / OUTPUT FILES ===
INPUT_CSV = "scan_downstream_pool.csv"
OUTPUT_CSV = INPUT_CSV.replace(".csv", "_multicontrast.csv")
LOG_FILE = INPUT_CSV.replace(".csv", "_multicontrast.log")

# === CONTRAST PRIORITY LIST ===
CONTRAST_ORDER = ["MPRAGE", "FLAIR", "T2", "IR", "GRE", "T1"]
MAX_CONTRASTS = 6

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV)

# === PRE-DEFINE COLUMN NAMES ===
base_cols = ["label", "subject_id", "date_acquired", "subject_sex", "subject_age", "subject_weight"]
contrast_cols = [f"{c.lower()}_path" for c in CONTRAST_ORDER[:MAX_CONTRASTS]]
bbox_cols = []
for c in CONTRAST_ORDER[:MAX_CONTRASTS]:
    bbox_cols += [
        f"{c.lower()}_xmin", f"{c.lower()}_xmax",
        f"{c.lower()}_ymin", f"{c.lower()}_ymax",
        f"{c.lower()}_zmin", f"{c.lower()}_zmax"
    ]

all_columns = base_cols + contrast_cols + bbox_cols

# === PROCESS DATA ===
multi_rows = []
log_lines = []

for (subject_id, date), group in df.groupby(["subject_id", "date_acquired"]):
    label = group["label"].iloc[0]
    sex = group["subject_sex"].iloc[0]
    age = group["subject_age"].iloc[0]
    weight = group["subject_weight"].iloc[0]

    row = {col: "" for col in all_columns}
    row.update({
        "label": label,
        "subject_id": subject_id,
        "date_acquired": date,
        "subject_sex": sex,
        "subject_age": age,
        "subject_weight": weight
    })

    used_contrasts = []
    bbox_data = []

    for contrast in CONTRAST_ORDER:
        scans = group[group["contrast"] == contrast]
        if not scans.empty and len(used_contrasts) < MAX_CONTRASTS:
            selected = scans.iloc[0]
            contrast_path_col = f"{contrast.lower()}_path"
            row[contrast_path_col] = selected["nii_file_path"]

            # Collect bbox data
            bbox_cols_group = [
                selected["xmin"], selected["xmax"],
                selected["ymin"], selected["ymax"],
                selected["zmin"], selected["zmax"]
            ]
            bbox_data.append((contrast.lower(), bbox_cols_group))

            used_contrasts.append(contrast)

    # if len(used_contrasts) < 2:
    #     continue  # Skip rows with fewer than 2 contrasts

        # Add bbox data at the end
    for idx, (contrast_name, bbox_vals) in enumerate(bbox_data):
        for i, col_name in enumerate(bbox_cols[idx*6:(idx+1)*6]):
            row[col_name] = bbox_vals[i]


    multi_rows.append(row)
    log_lines.append(f"{subject_id} on {date}: {', '.join(used_contrasts)}")

# === SAVE OUTPUT ===
out_df = pd.DataFrame(multi_rows, columns=all_columns)
out_df.to_csv(OUTPUT_CSV, index=False)

with open(LOG_FILE, "w") as f:
    f.write("Multi-contrast groupings (at most 6 per row):\n")
    f.write("\n".join(log_lines))

print(f"Wrote {len(out_df)} rows to {OUTPUT_CSV} and log to {LOG_FILE}")
