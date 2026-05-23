import pandas as pd
from collections import Counter
import os

# Paths
MASTER_CSV = "SCAN_NIFTI_all_with_cleaned_contrast.csv"
SPLITS = {
    "pretraining": "scan_pretraining.csv",
    "downstream_pool": "scan_downstream_pool.csv",
    "nc_vs_mci_train": "scan_downstream_nc_vs_mci_train.csv",
    "nc_vs_mci_val": "scan_downstream_nc_vs_mci_val.csv",
    "nc_vs_mci_test": "scan_downstream_nc_vs_mci_test.csv",
    "nc_vs_ad_train": "scan_downstream_nc_vs_ad_train.csv",
    "nc_vs_ad_val": "scan_downstream_nc_vs_ad_val.csv",
    "nc_vs_ad_test": "scan_downstream_nc_vs_ad_test.csv"
}
LOG_FILE = "scan_contrast_distribution_all_splits.log"

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

# Summary function
def summarize_contrast_distribution(df, name):
    log(f"\n[Contrast Distribution - {name}]")
    counts = Counter(df["contrast"])
    total = sum(counts.values())
    for contrast, count in counts.items():
        perc = 100 * count / total
        log(f"{contrast}: {count} scans ({perc:.2f}%)")

# Master file summary
df_master = pd.read_csv(MASTER_CSV)
summarize_contrast_distribution(df_master, "Master CSV")

# Split file summaries
for name, path in SPLITS.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        summarize_contrast_distribution(df, name)
    else:
        log(f"[WARNING] File not found: {path}")

# Write to log
with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))
log(f"\n[DONE] Full contrast summary written to {LOG_FILE}")
