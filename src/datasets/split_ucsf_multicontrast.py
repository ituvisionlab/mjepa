import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# --- Paths ---
MASTER_CSV = "ucsf_all_nii_multicontrast.csv"
LOG_FILE = "ucsf_multicontrast_split_summary.log"

GRADE_2_VS_34_SPLITS = {
    "train": "ucsf_multicontrast_grade2_vs_34_train.csv",
    "val": "ucsf_multicontrast_grade2_vs_34_val.csv",
    "test": "ucsf_multicontrast_grade2_vs_34_test.csv"
}

GRADE_23_VS_4_SPLITS = {
    "train": "ucsf_multicontrast_grade23_vs_4_train.csv",
    "val": "ucsf_multicontrast_grade23_vs_4_val.csv",
    "test": "ucsf_multicontrast_grade23_vs_4_test.csv"
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
    log(f"[LABEL DISTRIBUTION] {name}: total={total} → class 0: {counts[0]}, class 1: {counts[1]}")

# --- Load dataset ---
df = pd.read_csv(MASTER_CSV)
df["subject_id"] = df["subject_id"].astype(str)

# --- Grade 2 vs 3-4 Splitting ---
df_2_vs_34 = df[df["label"].isin([2, 3, 4])].copy()
df_2_vs_34["label"] = df_2_vs_34["label"].apply(lambda x: 0 if x == 2 else 1)

subjects_2_vs_34 = df_2_vs_34[["subject_id", "label"]].drop_duplicates()

train_ids, valtest_ids = train_test_split(
    subjects_2_vs_34,
    test_size=(val_ratio + test_ratio),
    stratify=subjects_2_vs_34["label"],
    random_state=random_state
)

val_ids, test_ids = train_test_split(
    valtest_ids,
    test_size=test_ratio / (val_ratio + test_ratio),
    stratify=valtest_ids["label"],
    random_state=random_state
)

for split, ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
    split_df = df_2_vs_34[df_2_vs_34["subject_id"].isin(ids["subject_id"])]
    split_df.to_csv(GRADE_2_VS_34_SPLITS[split], index=False)
    log(f"[INFO] Grade 2 vs 3-4 {split}: {len(split_df)} samples, {split_df['subject_id'].nunique()} subjects")
    log_class_distribution(f"Grade 2 vs 3-4 {split}", split_df)

# --- Grade 2-3 vs 4 Splitting ---
df_23_vs_4 = df[df["label"].isin([2, 3, 4])].copy()
df_23_vs_4["label"] = df_23_vs_4["label"].apply(lambda x: 1 if x == 4 else 0)

subjects_23_vs_4 = df_23_vs_4[["subject_id", "label"]].drop_duplicates()

train_ids, valtest_ids = train_test_split(
    subjects_23_vs_4,
    test_size=(val_ratio + test_ratio),
    stratify=subjects_23_vs_4["label"],
    random_state=random_state
)

val_ids, test_ids = train_test_split(
    valtest_ids,
    test_size=test_ratio / (val_ratio + test_ratio),
    stratify=valtest_ids["label"],
    random_state=random_state
)

for split, ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
    split_df = df_23_vs_4[df_23_vs_4["subject_id"].isin(ids["subject_id"])]
    split_df.to_csv(GRADE_23_VS_4_SPLITS[split], index=False)
    log(f"[INFO] Grade 2-3 vs 4 {split}: {len(split_df)} samples, {split_df['subject_id'].nunique()} subjects")
    log_class_distribution(f"Grade 2-3 vs 4 {split}", split_df)

# --- Write log file ---
with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))
log(f"[DONE] Log written to {LOG_FILE}")