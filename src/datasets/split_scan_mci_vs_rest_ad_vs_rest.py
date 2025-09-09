import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# --- File Paths ---
DOWNSTREAM_POOL_CSV = "scan_downstream_pool.csv"
LOG_FILE = "scan_downstream_additional_splits.log"

MCI_REST_SPLITS = {
    "train": "scan_downstream_mci_vs_rest_train.csv",
    "val": "scan_downstream_mci_vs_rest_val.csv",
    "test": "scan_downstream_mci_vs_rest_test.csv"
}
AD_REST_SPLITS = {
    "train": "scan_downstream_ad_vs_rest_train.csv",
    "val": "scan_downstream_ad_vs_rest_val.csv",
    "test": "scan_downstream_ad_vs_rest_test.csv"
}

# --- Parameters ---
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
random_state = 42

# --- Logging ---
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

def log_class_distribution(name, df):
    counts = Counter(df["label"])
    total = sum(counts.values())
    log(f"[LABEL DISTRIBUTION] {name}: total={total} → class 0: {counts.get(0,0)}, class 1: {counts.get(1,0)}")

# --- Load Downstream Pool ---
df = pd.read_csv(DOWNSTREAM_POOL_CSV)
df["subject_id"] = df["subject_id"].astype(str)

# --- MCI vs Rest (label 3 vs others) ---
mci_rest_df = df[df["label"].isin([1, 3, 4])].copy()
mci_rest_df["label"] = mci_rest_df["label"].map({3: 1, 1: 0, 4: 0})
mci_rest_subjects = mci_rest_df[["subject_id", "label"]].drop_duplicates()

train_ids, valtest_ids = train_test_split(
    mci_rest_subjects,
    test_size=(val_ratio + test_ratio),
    stratify=mci_rest_subjects["label"],
    random_state=random_state
)
val_ids, test_ids = train_test_split(
    valtest_ids,
    test_size=test_ratio / (val_ratio + test_ratio),
    stratify=valtest_ids["label"],
    random_state=random_state
)

for split, ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
    split_df = mci_rest_df[mci_rest_df["subject_id"].isin(ids["subject_id"])]
    split_df.to_csv(MCI_REST_SPLITS[split], index=False)
    log(f"[INFO] MCI vs Rest {split}: {len(split_df)} samples, {split_df['subject_id'].nunique()} subjects")
    log_class_distribution(f"MCI vs Rest {split}", split_df)

# --- AD vs Rest (label 4 vs others) ---
ad_rest_df = df[df["label"].isin([1, 3, 4])].copy()
ad_rest_df["label"] = ad_rest_df["label"].map({4: 1, 1: 0, 3: 0})
ad_rest_subjects = ad_rest_df[["subject_id", "label"]].drop_duplicates()

train_ids, valtest_ids = train_test_split(
    ad_rest_subjects,
    test_size=(val_ratio + test_ratio),
    stratify=ad_rest_subjects["label"],
    random_state=random_state
)
val_ids, test_ids = train_test_split(
    valtest_ids,
    test_size=test_ratio / (val_ratio + test_ratio),
    stratify=valtest_ids["label"],
    random_state=random_state
)

for split, ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
    split_df = ad_rest_df[ad_rest_df["subject_id"].isin(ids["subject_id"])]
    split_df.to_csv(AD_REST_SPLITS[split], index=False)
    log(f"[INFO] AD vs Rest {split}: {len(split_df)} samples, {split_df['subject_id'].nunique()} subjects")
    log_class_distribution(f"AD vs Rest {split}", split_df)

# --- Write Log ---
with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))
log(f"[DONE] Log written to {LOG_FILE}")
