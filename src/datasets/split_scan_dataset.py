import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# --- Paths ---
MASTER_CSV = "SCAN_NIFTI_all_with_betmask_and_bbox.csv"
PRETRAIN_CSV = "scan_pretraining.csv"
DOWNSTREAM_POOL_CSV = "scan_downstream_pool.csv"
LOG_FILE = "scan_split_summary.log"

MCI_SPLITS = {
    "train": "scan_downstream_nc_vs_mci_train.csv",
    "val": "scan_downstream_nc_vs_mci_val.csv",
    "test": "scan_downstream_nc_vs_mci_test.csv"
}
AD_SPLITS = {
    "train": "scan_downstream_nc_vs_ad_train.csv",
    "val": "scan_downstream_nc_vs_ad_val.csv",
    "test": "scan_downstream_nc_vs_ad_test.csv"
}

# --- Parameters ---
downstream_ratio = 0.25  # 25% of non-label2 subjects go to downstream
train_ratio = 0.7        # within downstream
val_ratio = 0.15
test_ratio = 0.15
random_state = 42

# --- Logging ---
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

def log_class_distribution(name, df):
    counts = Counter(df["binary_label"])
    total = sum(counts.values())
    log(f"[LABEL DISTRIBUTION] {name}: total={total} → class 0: {counts[0]}, class 1: {counts[1]}")

# --- Load dataset ---
df = pd.read_csv(MASTER_CSV)
df["subject_id"] = df["subject_id"].astype(str)

# --- Step 1: Split subjects ---
label2_subjects = df[df["label"] == 2]["subject_id"].unique()
other_subjects_df = df[df["label"].isin([1, 3, 4])][["subject_id", "label"]].drop_duplicates()
other_subject_ids = other_subjects_df["subject_id"].unique()

# --- Split non-label2 subjects ---
other_train_ids, downstream_ids = train_test_split(
    other_subject_ids,
    test_size=downstream_ratio,
    random_state=random_state
)

# --- Assign pretraining and downstream ---
pretrain_ids = set(label2_subjects) | set(other_train_ids)
pretrain_df = df[df["subject_id"].isin(pretrain_ids)].copy()
downstream_df = df[df["subject_id"].isin(downstream_ids)].copy()

pretrain_df.to_csv(PRETRAIN_CSV, index=False)
downstream_df.to_csv(DOWNSTREAM_POOL_CSV, index=False)

log(f"[INFO] Pretraining set: {len(pretrain_df)} samples, {pretrain_df['subject_id'].nunique()} subjects")
log(f"[INFO] Downstream pool: {len(downstream_df)} samples, {downstream_df['subject_id'].nunique()} subjects")

# --- Step 2a: NC vs MCI ---
mci_df = downstream_df[downstream_df["label"].isin([1, 3])].copy()
mci_df["binary_label"] = mci_df["label"].map({1: 0, 3: 1})
mci_subjects = mci_df[["subject_id", "binary_label"]].drop_duplicates()

train_ids, valtest_ids = train_test_split(
    mci_subjects,
    test_size=(val_ratio + test_ratio),
    stratify=mci_subjects["binary_label"],
    random_state=random_state
)
val_ids, test_ids = train_test_split(
    valtest_ids,
    test_size=test_ratio / (val_ratio + test_ratio),
    stratify=valtest_ids["binary_label"],
    random_state=random_state
)

for split, ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
    split_df = mci_df[mci_df["subject_id"].isin(ids["subject_id"])]
    split_df.to_csv(MCI_SPLITS[split], index=False)
    log(f"[INFO] NC vs MCI {split}: {len(split_df)} samples, {split_df['subject_id'].nunique()} subjects")
    log_class_distribution(f"NC vs MCI {split}", split_df)

# --- Step 2b: NC vs AD ---
ad_df = downstream_df[downstream_df["label"].isin([1, 4])].copy()
ad_df["binary_label"] = ad_df["label"].map({1: 0, 4: 1})
ad_subjects = ad_df[["subject_id", "binary_label"]].drop_duplicates()

train_ids, valtest_ids = train_test_split(
    ad_subjects,
    test_size=(val_ratio + test_ratio),
    stratify=ad_subjects["binary_label"],
    random_state=random_state
)
val_ids, test_ids = train_test_split(
    valtest_ids,
    test_size=test_ratio / (val_ratio + test_ratio),
    stratify=valtest_ids["binary_label"],
    random_state=random_state
)

for split, ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
    split_df = ad_df[ad_df["subject_id"].isin(ids["subject_id"])]
    split_df.to_csv(AD_SPLITS[split], index=False)
    log(f"[INFO] NC vs AD {split}: {len(split_df)} samples, {split_df['subject_id'].nunique()} subjects")
    log_class_distribution(f"NC vs AD {split}", split_df)

# --- Write log file ---
with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))
log(f"[DONE] Log written to {LOG_FILE}")
