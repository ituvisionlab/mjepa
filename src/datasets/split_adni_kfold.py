import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
# !!!CHECK and CHANGE the number of folds n_splits and output_dir

# === CONFIGURATION ===
file_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_nii_verified.csv'
output_dir = "adni_cv_5folds_stratified"
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
data_all = pd.read_csv(file_path)
data_all["subject_id"] = data_all["subject_id"].astype(str)

# Use only [0, 1, 2] for downstream logic
data_cv = data_all[data_all['label'].isin([0, 1, 2])].copy()

# === SUBJECT SUMMARY (INCLUDE ALL SUBJECTS FROM DATA_CV) ===
unique_subjects = data_cv[['subject_id', 'label']].drop_duplicates('subject_id')
subject_summary_full = unique_subjects.groupby('subject_id').agg(label=('label', lambda x: x.mode()[0])).reset_index()
subject_summary_full = subject_summary_full.sample(frac=1, random_state=42)

# === IDENTIFY LOW-VOLUME SUBJECTS FOR CV SPLITS (≤ 24 volumes) ===
volume_counts = data_cv.groupby('subject_id').size().reset_index(name='volume_count')
low_volume_subjects = volume_counts[volume_counts['volume_count'] <= 24]['subject_id']
cv_candidates = subject_summary_full[subject_summary_full['subject_id'].isin(low_volume_subjects)]

# === SELECT CV AND PRETRAIN SUBJECTS ===
cv_subject_summary = cv_candidates.iloc[:496]
pretrain_subject_summary = subject_summary_full[
    ~subject_summary_full['subject_id'].isin(cv_subject_summary['subject_id'])
]

# === SAVE SUBJECT SPLIT MAP ===
subject_summary = pd.concat([cv_subject_summary, pretrain_subject_summary])
cv_ids = set(cv_subject_summary['subject_id'])
subject_summary["split"] = subject_summary["subject_id"].apply(lambda sid: "cv" if sid in cv_ids else "pretrain")
subject_summary.to_csv(os.path.join(output_dir, "adni_subject_split_map.csv"), index=False)
print("✅ Saved subject split map.")

# === SAVE PRETRAIN AND DOWNSTREAM CSVs ===
cv_ids_all = set(cv_subject_summary['subject_id'])
pretrain_ids_all = set(data_all['subject_id']) - cv_ids_all

pretrain_data = data_all[data_all['subject_id'].isin(pretrain_ids_all)]
pretrain_data.to_csv(os.path.join(output_dir, "adni_pretrain.csv"), index=False)

downstream_data = data_all[data_all['subject_id'].isin(cv_ids_all) & data_all['label'].isin([0, 1, 2])]
downstream_data.to_csv(os.path.join(output_dir, "adni_downstream.csv"), index=False)

print(f"✅ Saved adni_pretrain.csv: {pretrain_data['subject_id'].nunique()} subjects, {len(pretrain_data)} volumes")
print(f"✅ Saved adni_downstream.csv: {downstream_data['subject_id'].nunique()} subjects, {len(downstream_data)} volumes")

# === FINAL SANITY CHECK ===
total_split_volume_count = len(pretrain_data) + len(downstream_data)
assert total_split_volume_count == len(data_all), f"⚠️ Volume mismatch: got {total_split_volume_count}, expected {len(data_all)}"
print(f"✅ Total volume accounting OK: {total_split_volume_count} volumes across pretrain + downstream")

# === SAVE SPLIT SUMMARY ===
split_summary = {
    "Split": ["Pretraining", "Downstream"],
    "Subjects": [pretrain_data['subject_id'].nunique(), downstream_data['subject_id'].nunique()],
    "Volumes": [len(pretrain_data), len(downstream_data)]
}
pd.DataFrame(split_summary).to_csv(os.path.join(output_dir, "adni_split_summary.csv"), index=False)
print("✅ Saved adni_split_summary.csv")

# === HELPER: Extract and optionally balance binary volumes ===
def extract_binary_split(data, subject_ids, label_pair, pos_label, balance_by_volume=False):
    df = data[data['subject_id'].isin(subject_ids) & data['label'].isin(label_pair)].copy()
    df['label'] = (df['label'] == pos_label).astype(int)

    if balance_by_volume and not df.empty:
        min_count = df['label'].value_counts().min()
        df = pd.concat([
            df[df['label'] == 0].sample(n=min_count, random_state=42),
            df[df['label'] == 1].sample(n=min_count, random_state=42)
        ])
        df = df.sort_values('subject_id').reset_index(drop=True)

    return df

# ===KFold SPLIT  ===
def generate_stratified_folds(task_name, label_pair, pos_label):
    task_subjects = cv_subject_summary.copy()

    n_splits=5
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=1881)
    subj_ids = task_subjects['subject_id'].values
    labels = task_subjects['label'].values

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(subj_ids, labels)):
        test_subjects = task_subjects.iloc[test_idx]
        trainval_subjects = task_subjects.iloc[trainval_idx]

        # === SPLIT TRAINVAL INTO 60% train / 20% val
        skf_inner = StratifiedKFold(n_splits, shuffle=True, random_state=fold + 100)
        train_idx, val_idx = next(skf_inner.split(trainval_subjects['subject_id'], trainval_subjects['label']))
        downtrain_subjects = trainval_subjects.iloc[train_idx]
        downval_subjects = trainval_subjects.iloc[val_idx]

        # === VALIDATE DOWNVAL SUBJECTS HAVE VOLUMES
        valid_downval_ids = []
        for sid in downval_subjects['subject_id']:
            vols = data_all[(data_all['subject_id'] == sid) & (data_all['label'].isin(label_pair))]
            if not vols.empty:
                valid_downval_ids.append(sid)
        downval_subjects = pd.DataFrame({'subject_id': valid_downval_ids})

        # === Save VAL and TEST
        for split, subjects_df in zip(["downval", "downtest"], [downval_subjects, test_subjects]):
            ids = subjects_df['subject_id'].values
            df = extract_binary_split(data_all, ids, label_pair, pos_label)
            out_path = os.path.join(output_dir, f"adni_fold{fold}_{split}_{task_name}.csv")
            df.to_csv(out_path, index=False)
            print(f"✅ Saved: {out_path} ({len(df)} volumes, {len(ids)} subjects)")
            print(df['label'].value_counts().to_dict())

        # === Save full downtrain (volume-balanced)
        train_ids = downtrain_subjects['subject_id'].values
        df = extract_binary_split(data_all, train_ids, label_pair, pos_label, balance_by_volume=True)
        df.to_csv(os.path.join(output_dir, f"adni_fold{fold}_downtrain_{task_name}.csv"), index=False)
        print(f"✅ Saved: fold{fold}_downtrain_{task_name} ({len(df)} volumes, {len(train_ids)} subjects)")
        print(df['label'].value_counts().to_dict())

# === RUN TASKS ===
generate_stratified_folds("nc_ad", label_pair=[0, 2], pos_label=2)
generate_stratified_folds("nc_mci", label_pair=[0, 1], pos_label=1)