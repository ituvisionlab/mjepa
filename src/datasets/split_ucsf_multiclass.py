import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter

# === CONFIGURATION ===
input_path = '/gpfs/home/unalg01/jepa/src/datasets/ucsf_all_nii.csv'
output_dir = '/gpfs/home/unalg01/jepa/src/datasets'
os.makedirs(output_dir, exist_ok=True)
log_path = os.path.join(output_dir, 'ucsf_split_summary_log_multiclass.txt')

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
random_seed = 1881
np.random.seed(random_seed)

log_lines = []
def log(text):
    print(text)
    log_lines.append(text)

def subject_wise_multiclass_split(df, split_name='ucsf_multiclass'):
    df_filtered = df[df['label'].isin([2, 3, 4])].copy()
    label_map = {2: 0, 3: 1, 4: 2}
    df_filtered['label'] = df_filtered['label'].map(label_map)

    # Get subject-wise unique labels
    subject_labels = df_filtered[['subject_id', 'label']].drop_duplicates()

    # Balance subject counts across all 3 classes
    min_count = subject_labels['label'].value_counts().min()
    balanced_subjects = pd.concat([
        subject_labels[subject_labels['label'] == label].sample(min_count, random_state=random_seed)
        for label in [0, 1, 2]
    ])

    # Split into Train / Val / Test
    trainval_subj, test_subj = train_test_split(
        balanced_subjects,
        test_size=(val_ratio + test_ratio),
        stratify=balanced_subjects['label'],
        random_state=random_seed
    )
    train_subj, val_subj = train_test_split(
        trainval_subj,
        test_size=(val_ratio / (train_ratio + val_ratio)),
        stratify=trainval_subj['label'],
        random_state=random_seed
    )

    def get_data(subj_df):
        return df_filtered[df_filtered['subject_id'].isin(subj_df['subject_id'])].copy()

    train_data = get_data(train_subj)
    val_data = get_data(val_subj)
    test_data = get_data(test_subj)

    total_subjects = len(balanced_subjects)
    total_volumes = len(df_filtered[df_filtered['subject_id'].isin(balanced_subjects['subject_id'])])

    log(f"==== {split_name.upper()} ====")
    for name, subset, subj_df in zip(['Train', 'Val', 'Test'],
                                     [train_data, val_data, test_data],
                                     [train_subj, val_subj, test_subj]):
        counts = Counter(subset['label'])
        subj_pct = len(subj_df) / total_subjects * 100
        vol_pct = len(subset) / total_volumes * 100
        log(f"{name}: {len(subset)} vols ({vol_pct:.2f}%), {len(subj_df)} subjects ({subj_pct:.2f}%), Class dist: {dict(counts)}")

    # Save splits
    train_data.to_csv(os.path.join(output_dir, f'{split_name}_train.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, f'{split_name}_val.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, f'{split_name}_test.csv'), index=False)

# === LOAD DATA ===
data = pd.read_csv(input_path)

# === SPLIT UCSF MULTICLASS ===
subject_wise_multiclass_split(data, split_name='ucsf_multiclass')

# === SAVE LOG ===
with open(log_path, 'w') as f:
    f.write("\n".join(log_lines))
log("UCSF 3-class balanced splits and logs saved.")
