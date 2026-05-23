#!/usr/bin/env python3
"""
scan_kfold_allsubjects.py

Creates subject-wise stratified k-fold splits for SCAN using ALL subjects.
This version matches the ADNI script selection logic and seeds:
 - shuffle seed (subject-level) = 42
 - outer StratifiedKFold random_state = 1881
 - inner StratifiedKFold random_state = fold + 100
 - CV candidate rule: subjects with <= 24 volumes
 - take first (shuffled) 496 candidates as CV (downstream)
 - pretrain = complement of those subjects

It also includes assertions to guarantee no subject overlap (no leakage).
"""

import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# === CONFIG ===
MASTER_CSV = "SCAN_NIFTI_all_filtered.csv"
OUTPUT_DIR = "scan_cv_5folds_stratified"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Canonical labelling used internally:
# 0 = ND / NC (normal control)
# 1 = MCI
# 2 = AD

N_SPLITS = 5
SUBJECT_SHUFFLE_SEED = 42      # same as second script sample seed
SKF_OUTER_SEED = 1881          # same as second script outer skf seed
BALANCE_TRAIN_BY_VOLUME = True

# desired CV candidate rules (same as ADNI logic)
VOLUME_THRESHOLD = 24
DESIRED_CV = 496

# Binary tasks (canonical label values)
TASKS = {
    "nc_ad":  {"label_pair": [0, 2], "pos_label": 2},  # NC vs AD  -> AD becomes 1
    "nc_mci": {"label_pair": [0, 1], "pos_label": 1}   # NC vs MCI -> MCI becomes 1
}

# === HELPERS ===
def safe_print(msg):
    print(msg)

def log_class_dist_from_series(series):
    c = Counter(series)
    return f"class0={c.get(0,0)}, class1={c.get(1,0)}"

def balance_volumes(df, seed=SUBJECT_SHUFFLE_SEED):
    if df.empty:
        return df
    vc = df['label'].value_counts()
    if len(vc) < 2:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = int(vc.min())
    df0 = df[df['label'] == 0].sample(n=n, random_state=seed)
    df1 = df[df['label'] == 1].sample(n=n, random_state=seed)
    out = pd.concat([df0, df1]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out

# === LOAD CSV ===
df_all = pd.read_csv(MASTER_CSV)
df_all["subject_id"] = df_all["subject_id"].astype(str).str.strip()
safe_print(f"[INFO] Loaded {MASTER_CSV}: {len(df_all)} rows, {df_all['subject_id'].nunique()} subjects")

# === REMAP / NORMALIZE LABELS to canonical 0/1/2 ===
if 'group' in df_all.columns:
    safe_print("[INFO] 'group' column found — mapping textual groups to canonical numeric labels.")
    text_to_label = {
        'ND': 0, 'NC': 0, 'Control': 0, 'CTL': 0, 'Normal': 0,
        'MCI': 1, 'mci': 1,
        'AD': 2, 'Alzheimer': 2, 'Alz': 2
    }
    df_all['label'] = df_all['group'].map(text_to_label)

unique_labels = sorted(pd.unique(df_all['label'].dropna()))
safe_print("[INFO] Detected label values (pre-normalize): " + str(unique_labels))

if set(unique_labels).issubset({0,1,2}):
    safe_print("[INFO] Labels already canonical {0,1,2}.")
else:
    alt_set = set([1,3,4])
    if alt_set.intersection(set(unique_labels)) and alt_set.issubset(set(unique_labels)):
        safe_print("[INFO] Detected labels {1,3,4} -> remapping to canonical {0,1,2}.")
        remap = {1:0, 3:1, 4:2}
        df_all['label'] = df_all['label'].map(remap).fillna(df_all['label'])
    else:
        safe_print("[WARN] Unexpected label set detected. Please inspect the label/group columns:")
        safe_print(df_all['label'].value_counts(dropna=False).to_string())

# Try cast to int if possible
try:
    df_all['label'] = df_all['label'].astype(int)
except Exception:
    pass

safe_print("[INFO] Final label values: " + str(sorted(pd.unique(df_all['label'].dropna()))))
safe_print("[INFO] Rows per label:\n" + df_all['label'].value_counts(dropna=False).to_string())
safe_print("[INFO] Unique subjects per label:\n" + df_all.groupby('label')['subject_id'].nunique().to_string())

# === SUBJECT SUMMARY (one row per subject: label by mode) ===
df_cv = df_all[df_all['label'].isin([0,1,2])].copy()
unique_subjects = df_cv[['subject_id', 'label']].drop_duplicates('subject_id')
subject_summary_full = unique_subjects.groupby('subject_id').agg(label=('label', lambda x: int(pd.Series(x).mode().iloc[0]))).reset_index()
# shuffle for randomness (use the same subject shuffle seed)
subject_summary_full = subject_summary_full.sample(frac=1, random_state=SUBJECT_SHUFFLE_SEED).reset_index(drop=True)
safe_print(f"[INFO] Built subject summary: {len(subject_summary_full)} subjects (shuffled)")

# === SELECT CV AND PRETRAIN SUBJECTS (ADNI logic) ===
# compute per-subject volume counts using df_cv (labels 0/1/2)
volume_counts = df_cv.groupby('subject_id').size().reset_index(name='volume_count')
volume_counts['subject_id'] = volume_counts['subject_id'].astype(str).str.strip()

# select low-volume subjects (<= VOLUME_THRESHOLD volumes)
low_volume_subjects = set(volume_counts[volume_counts['volume_count'] <= VOLUME_THRESHOLD]['subject_id'].values)
safe_print(f"[INFO] Found {len(low_volume_subjects)} subjects with <={VOLUME_THRESHOLD} volumes")

# restrict candidate pool to subject_summary_full subjects that are low-volume
cv_candidates = subject_summary_full[subject_summary_full['subject_id'].isin(low_volume_subjects)].reset_index(drop=True)
safe_print(f"[INFO] cv_candidates after intersecting with subject_summary_full: {len(cv_candidates)}")

# choose CV subjects (first DESIRED_CV after shuffle)
if len(cv_candidates) == 0:
    safe_print("[WARN] No cv_candidates found with <= threshold volumes. Falling back to using all subjects for CV.")
    cv_subject_summary = subject_summary_full.copy().reset_index(drop=True)
else:
    take_n = min(DESIRED_CV, len(cv_candidates))
    cv_subject_summary = cv_candidates.iloc[:take_n].reset_index(drop=True)
    if take_n < DESIRED_CV:
        safe_print(f"[WARN] Only {take_n} cv_candidates available (requested {DESIRED_CV}).")

# pretrain = complement of cv within subject_summary_full
pretrain_subject_summary = subject_summary_full[~subject_summary_full['subject_id'].isin(cv_subject_summary['subject_id'])].reset_index(drop=True)
safe_print(f"[INFO] CV subjects: {len(cv_subject_summary)}, Pretrain subjects: {len(pretrain_subject_summary)}")

# === SAFETY ASSERTS (prevent leakage) ===
cv_ids_set = set(cv_subject_summary['subject_id'])
pre_ids_set = set(pretrain_subject_summary['subject_id'])
assert len(cv_ids_set & pre_ids_set) == 0, f"LEAKAGE: {len(cv_ids_set & pre_ids_set)} subjects in both CV and pretrain!"

# ensure partition covers all subjects in subject_summary_full
all_subj_ids = set(subject_summary_full['subject_id'])
assert all_subj_ids == (cv_ids_set | pre_ids_set), "Subject partition mismatch: complement doesn't cover all subjects."

# Save subject split map
subject_summary = pd.concat([cv_subject_summary, pretrain_subject_summary], ignore_index=True)
subject_summary["split"] = subject_summary["subject_id"].apply(lambda sid: "cv" if sid in cv_ids_set else "pretrain")
subject_summary.to_csv(os.path.join(OUTPUT_DIR, "scan_subject_split_map.csv"), index=False)
safe_print(f"[INFO] Saved subject split map: {os.path.join(OUTPUT_DIR, 'scan_subject_split_map.csv')}")

# Save pretrain and downstream CSVs (only rows for subjects in each set)
pretrain_data = df_all[df_all['subject_id'].isin(pretrain_subject_summary['subject_id'])]
pretrain_path = os.path.join(OUTPUT_DIR, "scan_pretrain.csv")
pretrain_data.to_csv(pretrain_path, index=False)

downstream_data = df_all[df_all['subject_id'].isin(cv_subject_summary['subject_id'])]
downstream_path = os.path.join(OUTPUT_DIR, "scan_downstream.csv")
downstream_data.to_csv(downstream_path, index=False)

safe_print(f"[INFO] Saved pretrain CSV: {pretrain_path} ({len(pretrain_data)} rows, {pretrain_data['subject_id'].nunique()} subjects)")
safe_print(f"[INFO] Saved downstream CSV: {downstream_path} ({len(downstream_data)} rows, {downstream_data['subject_id'].nunique()} subjects)")

# Extra final assertions on saved CSV content
ds_ids = set(downstream_data['subject_id'].astype(str))
pt_ids = set(pretrain_data['subject_id'].astype(str))
assert ds_ids <= cv_ids_set, "downstream_data contains subjects not in cv_subject_summary"
assert pt_ids <= pre_ids_set, "pretrain_data contains subjects not in pretrain_subject_summary"
assert len(ds_ids & pt_ids) == 0, "LEAKAGE in saved CSVs: overlap between pretrain and downstream rows"

# === HELPER to extract and binarize volumes for a subject list ===
def extract_binary_split(data, subject_ids, label_pair, pos_label, balance_by_volume=False):
    df = data[data['subject_id'].isin(subject_ids) & data['label'].isin(label_pair)].copy()
    # Map to binary according to pos_label: positive -> 1, other -> 0
    df['label'] = (df['label'] == pos_label).astype(int)
    if balance_by_volume and not df.empty:
        df = balance_volumes(df, seed=SUBJECT_SHUFFLE_SEED)
    return df

# === K-FOLD SPLIT function (binary tasks) ===
def generate_stratified_folds_binary(task_name, label_pair, pos_label):
    logs = []
    # select subjects that belong to the task labels
    task_subjects = cv_subject_summary[cv_subject_summary['label'].isin(label_pair)].copy()
    if task_subjects.empty:
        safe_print(f"[WARN] No CV subjects for task {task_name} with labels {label_pair}")
        return

    # create subject-level binary label for stratification (1 if subject label == pos_label)
    task_subjects['bin_label'] = (task_subjects['label'] == pos_label).astype(int)
    safe_print(f"[INFO] Task {task_name}: {len(task_subjects)} subjects (subject-level) bin_label dist: {Counter(task_subjects['bin_label'])}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SKF_OUTER_SEED)
    subj_ids = task_subjects['subject_id'].values
    subj_bin_labels = task_subjects['bin_label'].values

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(subj_ids, subj_bin_labels)):
        test_subjects = task_subjects.iloc[test_idx].reset_index(drop=True)
        trainval_subjects = task_subjects.iloc[trainval_idx].reset_index(drop=True)

        # INNER SPLIT (nested): val = 1/5 of trainval -> ~16% overall
        skf_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SKF_OUTER_SEED + fold + 100)
        inner_gen = skf_inner.split(trainval_subjects['subject_id'], trainval_subjects['bin_label'])
        train_idx, val_idx = next(inner_gen)
        downtrain_subjects = trainval_subjects.iloc[train_idx].reset_index(drop=True)
        downval_subjects = trainval_subjects.iloc[val_idx].reset_index(drop=True)

        # Logging subject counts + percentages for clarity
        n_total = len(task_subjects)
        n_test = len(test_subjects)
        n_val = len(downval_subjects)
        n_train = len(downtrain_subjects)
        safe_print(f"[FOLD {fold}] subjects total={n_total} | train={n_train} ({n_train/n_total:.2%}) | "
                   f"val={n_val} ({n_val/n_total:.2%}) | test={n_test} ({n_test/n_total:.2%})")

        # ensure downval subjects have volumes for the task
        valid_downval_ids = [sid for sid in downval_subjects['subject_id'] if not df_all[(df_all['subject_id'] == sid) & (df_all['label'].isin(label_pair))].empty]
        downval_subjects = pd.DataFrame({'subject_id': valid_downval_ids})

        # Save downval and downtest (binary labels 0/1)
        for split_name, subjects_df in zip(["downval", "downtest"], [downval_subjects, test_subjects]):
            ids = subjects_df['subject_id'].values
            df_split = extract_binary_split(df_all, ids, label_pair, pos_label, balance_by_volume=False)
            out_path = os.path.join(OUTPUT_DIR, f"scan_fold{fold}_{split_name}_{task_name}.csv")
            df_split.to_csv(out_path, index=False)
            mapping_info = f"Mapping for {task_name}: original labels {label_pair} -> binary (pos_label={pos_label}): {label_pair[0]}->0, {pos_label}->1"
            logs.append(f"{task_name} fold{fold} {split_name}: rows={len(df_split)}, subjects={len(ids)}, {log_class_dist_from_series(df_split['label'])}, {mapping_info}")
            safe_print(f"[INFO] Saved {out_path}: rows={len(df_split)}, subjects={len(ids)}, {log_class_dist_from_series(df_split['label'])}; {mapping_info}")

        # Save downtrain (volume-balanced if enabled)
        train_ids = downtrain_subjects['subject_id'].values
        train_df = extract_binary_split(df_all, train_ids, label_pair, pos_label, balance_by_volume=BALANCE_TRAIN_BY_VOLUME)
        out_train = os.path.join(OUTPUT_DIR, f"scan_fold{fold}_downtrain_{task_name}.csv")
        train_df.to_csv(out_train, index=False)
        mapping_info = f"Mapping for {task_name}: original labels {label_pair} -> binary (pos_label={pos_label}): {label_pair[0]}->0, {pos_label}->1"
        logs.append(f"{task_name} fold{fold} downtrain: rows={len(train_df)}, subjects={len(train_ids)}, {log_class_dist_from_series(train_df['label'])}, {mapping_info}")
        safe_print(f"[INFO] Saved {out_train}: rows={len(train_df)}, subjects={len(train_ids)}, {log_class_dist_from_series(train_df['label'])}; {mapping_info}")

    # write task log
    log_path = os.path.join(OUTPUT_DIR, f"scan_fold_{task_name}_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(logs))
    safe_print(f"[INFO] Wrote log: {log_path}")

# === RUN TASKS ===
for task_name, params in TASKS.items():
    generate_stratified_folds_binary(task_name, params["label_pair"], params["pos_label"])

safe_print("[DONE] All tasks processed. Outputs in: " + OUTPUT_DIR)
