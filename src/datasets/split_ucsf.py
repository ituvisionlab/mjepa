import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter

# === CONFIGURATION ===
input_path = '/gpfs/home/unalg01/jepa/src/datasets/ucsf_all_nii.csv'
output_dir = '/gpfs/home/unalg01/jepa/src/datasets'
os.makedirs(output_dir, exist_ok=True)
log_path = os.path.join(output_dir, 'ucsf_split_summary_log.txt')

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
random_seed = 1881
np.random.seed(random_seed)

# === LOAD DATA ===
data = pd.read_csv(input_path)
log_lines = []

def log(text):
    print(text)
    log_lines.append(text)

def subject_wise_split(df, label_pair, split_name, k_shots=[8, 16, 32, 64]):
    df_filtered = df[df['label'].isin(label_pair)].copy()
    
    # Remap labels to [0, 1]
    label_mapping = {label_pair[0]: 0, label_pair[1]: 1}
    df_filtered['label'] = df_filtered['label'].map(label_mapping)

    subject_labels = df_filtered[['subject_id', 'label']].drop_duplicates()

    # Splitting
    trainval_subjects, test_subjects = train_test_split(
        subject_labels, test_size=(val_ratio + test_ratio),
        stratify=subject_labels['label'], random_state=random_seed
    )

    train_subjects, val_subjects = train_test_split(
        trainval_subjects, test_size=(val_ratio / (train_ratio + val_ratio)),
        stratify=trainval_subjects['label'], random_state=random_seed
    )

    def get_data(subj_df):
        return df_filtered[df_filtered['subject_id'].isin(subj_df['subject_id'])].copy()

    train_data = get_data(train_subjects)
    val_data = get_data(val_subjects)
    test_data = get_data(test_subjects)

    total_subjects = len(subject_labels)
    total_volumes = len(df_filtered)

    # === LOG STATS ===
    log(f"==== {split_name.upper()} ====")
    for name, subset, subj_df in zip(['Train', 'Val', 'Test'],
                                     [train_data, val_data, test_data],
                                     [train_subjects, val_subjects, test_subjects]):
        counts = Counter(subset['label'])
        subj_pct = len(subj_df) / total_subjects * 100
        vol_pct = len(subset) / total_volumes * 100
        log(f"{name}: {len(subset)} vols ({vol_pct:.2f}%), "
            f"{len(subj_df)} subjects ({subj_pct:.2f}%), "
            f"Class dist: {dict(counts)}")

    # === SAVE SPLITS ===
    train_data.to_csv(os.path.join(output_dir, f'{split_name}_train.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, f'{split_name}_val.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, f'{split_name}_test.csv'), index=False)

    # === FEW-SHOT SPLITS ===
    for k in k_shots:
        few_shot_parts = []
        for label in [0, 1]:
            label_subjects = train_subjects[train_subjects['label'] == label]
            n = min(k, len(label_subjects))
            sampled_subjects = label_subjects.sample(n=n, random_state=random_seed + k)
            few_shot_parts.append(sampled_subjects)

        few_shot_subjects = pd.concat(few_shot_parts)
        few_shot_data = get_data(few_shot_subjects)

        few_shot_data.to_csv(os.path.join(output_dir, f'{split_name}_train_k{k}.csv'), index=False)
        counts = Counter(few_shot_data['label'])
        log(f"Few-shot {k}/class: {len(few_shot_data)} vols, Class dist: {dict(counts)}")

# === UCSF SPLITS ===
# Grade 2 vs 3+4 → 2:0, 3&4:1
df_2_vs_34 = data[data['label'].isin([2, 3, 4])].copy()
df_2_vs_34['label'] = df_2_vs_34['label'].apply(lambda x: 0 if x == 2 else 1)
subject_wise_split(df_2_vs_34, label_pair=[0, 1], split_name='ucsf_grade2_vs_34')

# Grade 2+3 vs 4 → 2&3:0, 4:1
df_23_vs_4 = data[data['label'].isin([2, 3, 4])].copy()
df_23_vs_4['label'] = df_23_vs_4['label'].apply(lambda x: 1 if x == 4 else 0)
subject_wise_split(df_23_vs_4, label_pair=[0, 1], split_name='ucsf_grade23_vs_4')

# === SAVE LOG FILE ===
with open(log_path, 'w') as f:
    f.write("\n".join(log_lines))

log("All UCSF splits generated and logs saved.")
