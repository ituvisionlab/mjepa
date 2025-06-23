import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
input_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_cv_folds_stratified/adni_downstream.csv'
output_dir = '/gpfs/home/unalg01/jepa/src/datasets/adni_downstream_splits'
os.makedirs(output_dir, exist_ok=True)
log_path = os.path.join(output_dir, 'adni_split_summary_log.txt')
random_seed = 42  # reproducibility
np.random.seed(random_seed)

# === LOAD DATA ===
data = pd.read_csv(input_path)

log_lines = []

def log(text):
    print(text)
    log_lines.append(text)

def subject_wise_split(df, label_pair, pos_label, split_name, test_size=0.2, val_size=0.2, k_shots=[8, 16, 32, 64]):
    df_filtered = df[df['label'].isin(label_pair)].copy()
    
    # === IMPORTANT: Remap labels for CrossEntropyLoss ===
    # Map the labels to [0,1]:
    # For NC vs AD: NC=0 -> 0, AD=2 -> 1
    # For NC vs MCI: NC=0 -> 0, MCI=1 -> 1
    label_mapping = {label_pair[0]: 0, label_pair[1]: 1}
    df_filtered['label'] = df_filtered['label'].map(label_mapping)

    # Per subject binary label (one per subject)
    subject_labels = df_filtered[['subject_id', 'label']].drop_duplicates('subject_id')

    # Subject-wise Train/Test split
    trainval_subjects, test_subjects = train_test_split(
        subject_labels,
        test_size=test_size,
        stratify=subject_labels['label'],
        random_state=random_seed
    )

    # Subject-wise Train/Val split from TrainVal
    train_subjects, val_subjects = train_test_split(
        trainval_subjects,
        test_size=val_size,
        stratify=trainval_subjects['label'],
        random_state=random_seed
    )

    def get_data_for_subjects(subj_df):
        return df_filtered[df_filtered['subject_id'].isin(subj_df['subject_id'])].copy()

    train_data = get_data_for_subjects(train_subjects)
    val_data   = get_data_for_subjects(val_subjects)
    test_data  = get_data_for_subjects(test_subjects)

    total_subjects = len(subject_labels)
    total_volumes  = len(df_filtered)

    # === LOG STATS ===
    log(f"==== {split_name.upper()} ====")
    for name, subset, subj_df in zip(['Train', 'Val', 'Test'], [train_data, val_data, test_data], [train_subjects, val_subjects, test_subjects]):
        counts = subset['label'].value_counts().to_dict()
        subject_pct = len(subj_df) / total_subjects * 100
        volume_pct  = len(subset) / total_volumes * 100
        log(f"{name}: {len(subset)} volumes ({volume_pct:.2f}%), {len(subj_df)} subjects ({subject_pct:.2f}%), Class dist: {counts}")

    # === SAVE SPLITS ===
    train_data.to_csv(os.path.join(output_dir, f'{split_name}_train.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, f'{split_name}_val.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, f'{split_name}_test.csv'), index=False)

    # === FEW-SHOT SPLITS FROM TRAIN ===
    for k in k_shots:
        few_shot_parts = []
        for label in [0, 1]:
            label_subjects = train_subjects[train_subjects['label'] == label]
            n = min(k, len(label_subjects))
            sampled_subjects = label_subjects.sample(n=n, random_state=random_seed + k)
            few_shot_parts.append(sampled_subjects)

        few_shot_subjects = pd.concat(few_shot_parts)
        few_shot_data = get_data_for_subjects(few_shot_subjects)

        few_shot_path = os.path.join(output_dir, f'{split_name}_train_k{k}.csv')
        few_shot_data.to_csv(few_shot_path, index=False)

        counts = few_shot_data['label'].value_counts().to_dict()
        log(f"Few-shot {k} per class: {len(few_shot_data)} volumes, Class dist: {counts}")

# === NC vs AD ===
subject_wise_split(
    df=data,
    label_pair=[0, 2],   # NC vs AD
    pos_label=2,
    split_name='nc_ad'
)

# === NC vs MCI ===
subject_wise_split(
    df=data,
    label_pair=[0, 1],   # NC vs MCI
    pos_label=1,
    split_name='nc_mci'
)

# === SAVE LOG FILE ===
with open(log_path, 'w') as f:
    for line in log_lines:
        f.write(line + '\n')

log("✅ All splits and logs saved.")
