import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from datetime import datetime

# === CONFIG ===
csv_path = "/gpfs/home/unalg01/jepa/src/datasets/adni_cv_folds_stratified/adni_fold3_downtrain_nc_ad.csv"
log_path = "fold3_check_python.log"

# --- Setup dual logging (to screen and file) ---
class TeeLogger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w", buffering=1)  # line-buffered
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger(log_path)
sys.stderr = sys.stdout

print(f"\n[{datetime.now()}] 🔍 Starting fold check for: {csv_path}\n")

# --- Load CSV ---
if not os.path.isfile(csv_path):
    print(f"❌ CSV file not found: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"✅ Loaded CSV with {len(df)} rows.\n")

# --- Label distribution ---
print("Label distribution:")
print(df["label"].value_counts(dropna=False))
print("\n")

# --- Initialize containers ---
missing_files, corrupt_files, zero_vols, nan_vols = [], [], [], []
means, stds = [], []

for idx, row in df.iterrows():
    path = row["nii_file_path"]
    label = row["label"]

    if not os.path.isfile(path):
        missing_files.append(path)
        continue

    try:
        img = nib.load(path)
        data = img.get_fdata()

        if np.isnan(data).any() or np.isinf(data).any():
            nan_vols.append(path)
            continue

        mean, std = data.mean(), data.std()
        if std == 0 or np.allclose(std, 0):
            zero_vols.append(path)

        means.append(mean)
        stds.append(std)

    except Exception as e:
        corrupt_files.append((path, str(e)))
        continue

# --- Summary ---
print("\n=== SUMMARY ===")
print(f"✅ Loaded successfully: {len(means)}")
print(f"❌ Missing files: {len(missing_files)}")
print(f"⚠️ Corrupt NIfTI headers: {len(corrupt_files)}")
print(f"⚠️ All-zero/constant volumes: {len(zero_vols)}")
print(f"⚠️ NaN/Inf volumes: {len(nan_vols)}")

if means:
    print(f"\n📊 Mean intensity: {np.mean(means):.3f} ± {np.std(means):.3f}")
    print(f"📊 Std intensity: {np.mean(stds):.3f} ± {np.std(stds):.3f}")

# --- Problem examples ---
def print_examples(title, items, limit=5):
    if items:
        print(f"\n{title} (showing {min(limit, len(items))}):")
        for x in items[:limit]:
            if isinstance(x, tuple): print(f"{x[0]} -> {x[1]}")
            else: print(x)

print_examples("Missing file examples", missing_files)
print_examples("Corrupt file examples", corrupt_files)
print_examples("Constant-volume examples", zero_vols)
print_examples("NaN/Inf volume examples", nan_vols)

print(f"\n[{datetime.now()}] ✅ Check complete.")
print(f"Results saved to {os.path.abspath(log_path)}\n")
