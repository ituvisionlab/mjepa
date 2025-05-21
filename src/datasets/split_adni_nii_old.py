import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split

# === CONFIGURATION ===
file_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_nii_verified.csv'   # "adni_all_bet_nii.csv"
output_dir = "adni_cv_folds_stratified"
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
data = pd.read_csv(file_path)
data = data[data['label'].isin([0, 1, 2])]  # Use only NC, MCI, AD

# === SUBJECT SUMMARY ===
subject_summary = data.groupby('subject_id').agg(
    label=('label', lambda x: x.mode()[0])
).reset_index().sample(frac=1, random_state=42)

# === SELECT 496 SUBJECTS FOR DOWNSTREAM ===
cv_subjects = subject_summary.iloc[:496]
pretrain_subjects = subject_summary.iloc[496:]

# Save pretrain and downstream reference files
data[data['subject_id'].isin(pretrain_subjects['subject_id'])].to_csv(
    os.path.join(output_dir, "adni_pretrain.csv"), index=False
)
data[data['subject_id'].isin(cv_subjects['subject_id'])].to_csv(
    os.path.join(output_dir, "adni_downstream.csv"), index=False
)

# === HELPER TO EXTRACT + RELABEL FOR BINARY TASK ===
def extract_binary_split(data, subject_ids, label_pair, pos_label):
    df = data[data['subject_id'].isin(subject_ids) & data['label'].isin(label_pair)].copy()
    df['label'] = (df['label'] == pos_label).astype(int)
    return df

# === MAIN FUNCTION TO GENERATE CV FOLDS ===
def generate_stratified_folds(task_name, label_pair, pos_label):
    task_subjects = cv_subjects[cv_subjects['label'].isin(label_pair)].copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    subj_ids = task_subjects['subject_id'].values
    labels = task_subjects['label'].values

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(subj_ids, labels)):
        test_subjects = task_subjects.iloc[test_idx]
        trainval_subjects = task_subjects.iloc[trainval_idx]

        # Sample ~128 subjects per class for training
        downtrain_parts = []
        remaining_subjects = trainval_subjects.copy()

        for label in label_pair:
            candidates = trainval_subjects[trainval_subjects['label'] == label]
            n = min(128, len(candidates))
            sampled = candidates.sample(n=n, random_state=fold)
            downtrain_parts.append(sampled)
            remaining_subjects = remaining_subjects[~remaining_subjects['subject_id'].isin(sampled['subject_id'])]

        downtrain_df = pd.concat(downtrain_parts).reset_index(drop=True)
        downval_df = remaining_subjects.reset_index(drop=True)
        downtest_df = test_subjects.reset_index(drop=True)

        # Save CSVs for this fold
        for split, subjects_df in zip(
            ["downtrain", "downval", "downtest"],
            [downtrain_df, downval_df, downtest_df]
        ):
            ids = subjects_df['subject_id'].values
            df = extract_binary_split(data, ids, label_pair, pos_label)
            out_path = os.path.join(output_dir, f"adni_fold{fold}_{split}_{task_name}.csv")
            df.to_csv(out_path, index=False)
            print(f"✅ Saved: {out_path} ({len(df)} volumes, {len(ids)} subjects)")

# === GENERATE FOR BOTH TASKS ===
generate_stratified_folds("nc_ad", label_pair=[0, 2], pos_label=2)
generate_stratified_folds("nc_mci", label_pair=[0, 1], pos_label=1)

# === SUBJECT MAP CSV ===
subject_map_df = pd.DataFrame({
    "subject_id": subject_summary["subject_id"],
    "label": subject_summary["label"],
    "split": ["cv"] * 496 + ["pretrain"] * (len(subject_summary) - 496)
})
subject_map_df.to_csv(os.path.join(output_dir, "adni_subject_split_map.csv"), index=False)
