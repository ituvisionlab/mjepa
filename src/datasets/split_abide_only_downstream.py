import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import os

# === Config ===
CSV_PATH = "/gpfs/home/unalg01/jepa/src/datasets/ABIDE_master_with_betmask.csv"
OUTPUT_DIR = "/gpfs/home/unalg01/jepa/src/datasets"
#os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_PATHS = {
    "train": os.path.join(OUTPUT_DIR, "abide_downtrain.csv"),
    "val": os.path.join(OUTPUT_DIR, "abide_downval.csv"),
    "test": os.path.join(OUTPUT_DIR, "abide_downtest.csv"),
}
LOG_FILE = os.path.join(OUTPUT_DIR, "abide_downstream_split_summary.log")

# === Parameters ===
train_ratio = 0.70
val_ratio = 0.12
test_ratio = 0.18
random_state = 42

# === Load dataset ===
df = pd.read_csv(CSV_PATH)
df["subject_id"] = df["subject_id"].astype(str)

# === Log helper ===
log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

def log_distribution(name, subset_df):
    counts = Counter(subset_df["label"])
    total = sum(counts.values())
    log(f"[{name}] Total: {total} → Class 0 (Control): {counts[0]}, Class 1 (Autism): {counts[1]}")

# === Subject-level stratified split ===
subjects = df[["subject_id", "label"]].drop_duplicates()

train_subj, valtest_subj = train_test_split(
    subjects,
    test_size=(val_ratio + test_ratio),
    stratify=subjects["label"],
    random_state=random_state
)

val_subj, test_subj = train_test_split(
    valtest_subj,
    test_size=test_ratio / (val_ratio + test_ratio),
    stratify=valtest_subj["label"],
    random_state=random_state
)

splits = {
    "train": df[df["subject_id"].isin(train_subj["subject_id"])],
    "val": df[df["subject_id"].isin(val_subj["subject_id"])],
    "test": df[df["subject_id"].isin(test_subj["subject_id"])],
}

# === Save splits and log ===
for name, split_df in splits.items():
    split_df.to_csv(OUT_PATHS[name], index=False)
    log(f"[SPLIT] {name}: {len(split_df)} scans, {split_df['subject_id'].nunique()} subjects")
    log_distribution(name, split_df)

with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))
log(f"[DONE] Split summary saved to {LOG_FILE}")
