import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from datetime import datetime

# === CONFIG ===
base_dir = "/gpfs/home/unalg01/jepa/src/datasets/adni_cv_5folds_stratified"
fold_paths = [
    os.path.join(base_dir, f"adni_fold{i}_downtrain_nc_mci.csv") for i in range(5)
]

# --- Setup dual logging (screen + file) ---
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

# --- Function to check one fold ---
def check_fold(csv_path):
    fold_id = os.path.basename(csv_path).split("_")[1]
    log_path = f"fold{fold_id}_check.log"

    sys.stdout = TeeLogger(log_path)
    sys.stderr = sys.stdout

    print(f"\n[{datetime.now()}] 🔍 Starting fold check for: {csv_path}\n")

    if not os.path.isfile(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return {"fold": fold_id, "status": "missing_csv"}

    df = pd.read_csv(csv_path)
    print(f"✅ Loaded CSV with {len(df)} rows.\n")

    print("Label distribution:")
    print(df["label"].value_counts(dropna=False))
    print("\n")

    missing_files, corrupt_files, zero_vols, nan_vols = [], [], [], []
    means, stds = [], []

    for idx, row in df.iterrows():
        path = row["nii_file_path"]

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

    print("\n=== SUMMARY ===")
    print(f"✅ Loaded successfully: {len(means)}")
    print(f"❌ Missing files: {len(missing_files)}")
    print(f"⚠️ Corrupt NIfTI headers: {len(corrupt_files)}")
    print(f"⚠️ All-zero/constant volumes: {len(zero_vols)}")
    print(f"⚠️ NaN/Inf volumes: {len(nan_vols)}")

    if means:
        print(f"\n📊 Mean intensity: {np.mean(means):.3f} ± {np.std(means):.3f}")
        print(f"📊 Std intensity: {np.mean(stds):.3f} ± {np.std(stds):.3f}")

    print(f"\n[{datetime.now()}] ✅ Fold {fold_id} check complete.")
    print(f"Results saved to {os.path.abspath(log_path)}\n")

    # Return a compact summary for terminal view
    return {
        "fold": fold_id,
        "rows": len(df),
        "missing": len(missing_files),
        "corrupt": len(corrupt_files),
        "zero_vol": len(zero_vols),
        "nan_vol": len(nan_vols),
        "mean_intensity": np.mean(means) if means else np.nan,
        "std_intensity": np.mean(stds) if stds else np.nan,
    }

# --- Run all folds ---
results = []
for csv_path in fold_paths:
    print(f"\n=== Checking {csv_path} ===\n")
    res = check_fold(csv_path)
    results.append(res)

# --- Restore normal stdout ---
sys.stdout = sys.__stdout__

# --- Summary Table ---
summary_df = pd.DataFrame(results)
print("\n\n===== FINAL SUMMARY ACROSS FOLDS =====\n")
print(summary_df.to_string(index=False))
summary_df.to_csv("fold_integrity_summary_adni_nc_mci.csv", index=False)
print("\n✅ Saved overall summary to fold_integrity_summary.csv\n")
