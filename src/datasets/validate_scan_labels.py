import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser

# Paths
MASTER_CSV = "SCAN_NIFTI_all_with_betmask_and_bbox.csv"
META_CSV = "/gpfs/data/sodicksonlab/gozde/SCAN/investigator_nacc69.csv"
LOG_FILE = "scan_label_validation_report.log"

# Load master CSV (with labels assigned)
df = pd.read_csv(MASTER_CSV)
df["subject_id"] = df["subject_id"].astype(str)
df["date_acquired"] = pd.to_datetime(df["date_acquired"], errors="coerce")

# Load metadata with relevant fields
meta_cols = ["NACCID", "NACCUDSD", "VISITYR", "VISITMO", "VISITDAY"]
meta = pd.read_csv(META_CSV, usecols=meta_cols, dtype={"NACCID": str})
meta["VISITYR"] = pd.to_numeric(meta["VISITYR"], errors="coerce")
meta["VISITMO"] = pd.to_numeric(meta["VISITMO"], errors="coerce")
meta["VISITDAY"] = pd.to_numeric(meta["VISITDAY"], errors="coerce")

# Create visit date column
meta = meta.dropna(subset=["VISITYR", "VISITMO", "VISITDAY"])
meta["visit_date"] = pd.to_datetime(dict(year=meta["VISITYR"], month=meta["VISITMO"], day=meta["VISITDAY"]), errors="coerce")

# Prepare log
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

# Validation loop
n_total = 0
n_matched = 0
n_mismatch = 0
n_missing = 0

for idx, row in df.iterrows():
    subject = row["subject_id"]
    scan_date = row["date_acquired"]
    assigned_label = row["label"]

    # Find subject in metadata
    subject_meta = meta[meta["NACCID"] == subject]
    if subject_meta.empty or pd.isna(scan_date):
        n_missing += 1
        continue

    # Find closest metadata visit date
    subject_meta = subject_meta.copy()
    subject_meta["delta"] = (subject_meta["visit_date"] - scan_date).abs()
    closest = subject_meta.sort_values("delta").iloc[0]
    metadata_label = closest["NACCUDSD"]

    if pd.isna(metadata_label):
        n_missing += 1
        continue

    n_total += 1
    if int(assigned_label) == int(metadata_label):
        n_matched += 1
    else:
        n_mismatch += 1
        log(f"Mismatch: subject={subject}, scan={scan_date.date()}, assigned={assigned_label}, meta={metadata_label}")

# Summary
log("\n--- Validation Summary ---")
log(f"Total validated: {n_total}")
log(f"Correct matches: {n_matched}")
log(f"Mismatches: {n_mismatch}")
log(f"Missing metadata: {n_missing}")

# Write log
with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))
log(f"[DONE] Validation log written to {LOG_FILE}")
