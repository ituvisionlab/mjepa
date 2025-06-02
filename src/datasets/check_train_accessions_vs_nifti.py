import os
import pandas as pd

# --- CONFIG ---
nifti_root = "/gpfs/data/prostatelab/NIFTI"
bx_train_csv = "/gpfs/data/prostatelab/NIFTI_csv/Bx_train_set_2025_11_4.csv"
log_missing = "/gpfs/home/unalg01/jepa/src/datasets/missing_train_subjects.log"

# --- LOAD Bx TRAIN SET ---
df = pd.read_csv(bx_train_csv, dtype=str)
accessions = df["AccessionNumber"].astype(str).str.strip().str.split(".").str[0]
accessions_set = set(accessions)

print(f"✅ Loaded {len(accessions_set)} accession numbers from Bx train CSV")

# --- GET FOLDER NAMES IN NIFTI DIR ---
nifti_folders = set(
    s.strip().split(".")[0] for s in os.listdir(nifti_root)
    if os.path.isdir(os.path.join(nifti_root, s))
)

print(f"✅ Found {len(nifti_folders)} subject folders in {nifti_root}")

# --- COMPARE ---
found = sorted(accessions_set & nifti_folders)
missing = sorted(accessions_set - nifti_folders)

print(f"✅ Match: {len(found)}")
print(f"❌ Missing: {len(missing)}")

# --- OPTIONAL: SAVE MISSING LIST ---
with open(log_missing, 'w') as f:
    for acc in missing:
        f.write(acc + "\n")

print(f"📝 Missing subject IDs written to {log_missing}")
