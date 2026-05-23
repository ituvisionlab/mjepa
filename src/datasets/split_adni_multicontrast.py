import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
input_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_cv_folds_stratified/adni_downstream_multicontrast.csv'
output_dir = '/gpfs/home/unalg01/jepa/src/datasets/adni_downstream_multicontrast_splits'
os.makedirs(output_dir, exist_ok=True)
log_path = os.path.join(output_dir, 'adni_multicontrast_split_summary_log.txt')
random_seed = 42  # reproducibility
np.random.seed(random_seed)

# === LOAD DATA ===
data = pd.read_csv(input_path)

log_lines = []

def log(text):
    print(text)
    log_lines.append(text)

def subject_wise_split(df, split_name, test_size=0.2, val_size=0.2):
    # Per subject binary label (one per subject)
    subject_labels = df[['subject_id', 'label']].drop_duplicates('subject_id')

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
        return df[df['subject_id'].isin(subj_df['subject_id'])].copy()

    train_data = get_data_for_subjects(train_subjects)
    val_data   = get_data_for_subjects(val_subjects)
    test_data  = get_data_for_subjects(test_subjects)

    total_subjects = len(subject_labels)
    total_samples  = len(df)

    # === LOG STATS ===
    log(f"==== {split_name.upper()} ====")
    for name, subset, subj_df in zip(['Train', 'Val', 'Test'], [train_data, val_data, test_data], [train_subjects, val_subjects, test_subjects]):
        counts = subset['label'].value_counts().to_dict()
        subject_pct = len(subj_df) / total_subjects * 100
        sample_pct  = len(subset) / total_samples * 100
        log(f"{name}: {len(subset)} samples ({sample_pct:.2f}%), {len(subj_df)} subjects ({subject_pct:.2f}%), Class dist: {counts}")

    # === SAVE SPLITS ===
    train_data.to_csv(os.path.join(output_dir, f'{split_name}_train.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, f'{split_name}_val.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, f'{split_name}_test.csv'), index=False)

# === NC vs AD ===
df_nc_ad = data[data['label'].isin([0, 2])].copy()
df_nc_ad['label'] = df_nc_ad['label'].map({0: 0, 2: 1})
subject_wise_split(
    df=df_nc_ad,
    split_name='nc_ad'
)

# === NC vs MCI ===
df_nc_mci = data[data['label'].isin([0, 1])].copy()
df_nc_mci['label'] = df_nc_mci['label'].map({0: 0, 1: 1})
subject_wise_split(
    df=df_nc_mci,
    split_name='nc_mci'
)

# === SAVE LOG FILE ===
with open(log_path, 'w') as f:
    for line in log_lines:
        f.write(line + '\n')

log("All splits and logs saved.")
