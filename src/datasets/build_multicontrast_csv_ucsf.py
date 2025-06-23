import pandas as pd
import os

# === INPUT / OUTPUT FILES ===
INPUT_CSV = "ucsf_all_nii.csv"
OUTPUT_CSV = INPUT_CSV.replace(".csv", "_multicontrast.csv")
LOG_FILE = INPUT_CSV.replace(".csv", "_multicontrast.log")

# === CONTRAST PRIORITY LIST ===
CONTRAST_ORDER = ["T1", "T1c", "FLAIR", "T2", "ADC", "DWI"]
MAX_CONTRASTS = 6

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV)

multi_rows = []
log_lines = []

# === GROUP BY SUBJECT ===
for subject_id, group in df.groupby("subject_id"):
    label = group["label"].iloc[0]
    sex = group["subject_sex"].iloc[0]
    age = group["subject_age"].iloc[0]
    weight = group["subject_weight"].iloc[0]

    selected_contrasts = {}
    
    # explicit logic for contrasts
    for contrast in CONTRAST_ORDER:
        if contrast == "T1":
            scans = group[group["contrast"].str.upper() == "T1"]
        elif contrast == "T1c":
            scans = group[group["nii_file_path"].str.contains("T1c", case=False)]
        else:
            scans = group[group["contrast"].str.upper() == contrast.upper()]

        if not scans.empty:
            selected_contrasts[contrast] = scans.iloc[0]

    if len(selected_contrasts) == 0:
        continue

    row = {
        "label": label,
        "subject_id": subject_id,
        "subject_sex": sex,
        "subject_age": age,
        "subject_weight": weight,
    }

    # Add contrast paths first
    for contrast in CONTRAST_ORDER:
        key = f"{contrast.lower()}_path"
        row[key] = selected_contrasts[contrast]["nii_file_path"] if contrast in selected_contrasts else ""

    # Then bbox columns
    for contrast in CONTRAST_ORDER:
        if contrast in selected_contrasts:
            entry = selected_contrasts[contrast]
            base = contrast.lower()
            row[f"{base}_xmin"] = entry["xmin"]
            row[f"{base}_xmax"] = entry["xmax"]
            row[f"{base}_ymin"] = entry["ymin"]
            row[f"{base}_ymax"] = entry["ymax"]
            row[f"{base}_zmin"] = entry["zmin"]
            row[f"{base}_zmax"] = entry["zmax"]
        else:
            base = contrast.lower()
            row[f"{base}_xmin"] = row[f"{base}_xmax"] = ""
            row[f"{base}_ymin"] = row[f"{base}_ymax"] = ""
            row[f"{base}_zmin"] = row[f"{base}_zmax"] = ""

    multi_rows.append(row)
    log_lines.append(f"{subject_id}: {', '.join(selected_contrasts.keys())}")

# === SAVE OUTPUT ===
out_df = pd.DataFrame(multi_rows)
out_df.to_csv(OUTPUT_CSV, index=False)

with open(LOG_FILE, "w") as f:
    f.write(f"Wrote {len(out_df)} rows to {OUTPUT_CSV} and log to {LOG_FILE}\n")
    f.write("Multi-contrast groupings (at most 6 per row):\n")
    f.write("\n".join(log_lines))

print(f"Wrote {len(out_df)} rows to {OUTPUT_CSV} and log to {LOG_FILE}")
