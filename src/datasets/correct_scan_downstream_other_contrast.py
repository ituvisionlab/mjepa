import pandas as pd

INPUT_CSV = "scan_downstream_pool.csv"
OUTPUT_CSV = "scan_downstream_pool_corrected.csv"
LOG_FILE = "scan_contrast_fix.log"

# Load the CSV
df = pd.read_csv(INPUT_CSV)

# Track how many were fixed
corrections = {
    "OTHER→T1": 0,
    "OTHER→MPRAGE": 0
}

# Fix contrasts
def fix_contrast(row):
    if row["contrast"] == "OTHER":
        path = str(row["nii_file_path"]).lower()
        if "t1" in path:
            corrections["OTHER→T1"] += 1
            return "T1"
        elif "mpr" in path:
            corrections["OTHER→MPRAGE"] += 1
            return "MPRAGE"
    return row["contrast"]

df["contrast"] = df.apply(fix_contrast, axis=1)

# Write corrected CSV
df.to_csv(OUTPUT_CSV, index=False)

# Write log
with open(LOG_FILE, "w") as f:
    f.write("Correction Summary:\n")
    for k, v in corrections.items():
        f.write(f"{k}: {v}\n")
    f.write(f"Total rows corrected: {sum(corrections.values())}\n")
    remaining = (df["contrast"] == "OTHER").sum()
    f.write(f"Remaining 'OTHER' rows: {remaining}\n")

print(f"✅ Done. Corrected CSV written to {OUTPUT_CSV} and log to {LOG_FILE}")

