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

log_lines = []
def log(text):
    print(text)
    log_lines.append(text)

def subject_wise_balanced_split(df, label_pair, label_map, split_name, k_shots=[8, 16, 32, 64]):
    df_filtered = df[df['label'].isin(label_pair)].copy()
    df_filtered['label'] = df_filtered['label'].map(label_map)

    # Subject-level label
    subject_labels = df_filtered[['subject_id', 'label']].drop_duplicates()
    
    # Balance classes at subject level
    grouped = subject_labels.groupby('label')
    min_class_size = grouped.size().min()
    balanced_subjects = grouped.apply(lambda g: g.sample(n=min_class_size, random_state=random_seed)).reset_index(drop=True)

    # Split into train/val/test
    trainval_subj, test_subj = train_test_split(
        balanced_subjects, test_size=(val_ratio + test_ratio),
        stratify=balanced_subjects['label'], random_state=random_seed
    )
    train_subj, val_subj = train_test_split(
        trainval_subj, test_size=(val_ratio / (train_ratio + val_ratio)),
        stratify=trainval_subj['label'], random_state=random_seed
    )

    def get_data(subj_df):
        return df_filtered[df_filtered['subject_id'].isin(subj_df['subject_id'])].copy()

    train_data = get_data(train_subj)
    val_data   = get_data(val_subj)
    test_data  = get_data(test_subj)

    total_subjects = len(balanced_subjects)
    total_volumes  = len(df_filtered[df_filtered['subject_id'].isin(balanced_subjects['subject_id'])])

    log(f"==== {split_name.upper()} ====")
    for name, subset, subj_df in zip(['Train', 'Val', 'Test'],
                                     [train_data, val_data, test_data],
                                     [train_subj, val_subj, test_subj]):
        counts = Counter(subset['label'])
        subj_pct = len(subj_df) / total_subjects * 100
        vol_pct = len(subset) / total_volumes * 100
        log(f"{name}: {len(subset)} vols ({vol_pct:.2f}%), {len(subj_df)} subjects ({subj_pct:.2f}%), Class dist: {dict(counts)}")

    # Save full splits
    train_data.to_csv(os.path.join(output_dir, f'{split_name}_train.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, f'{split_name}_val.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, f'{split_name}_test.csv'), index=False)

    # === FEW-SHOT SPLITS ===
    for k in k_shots:
        few_shot_parts = []
        for label in [0, 1]:
            label_subj = train_subj[train_subj['label'] == label]
            n = min(k, len(label_subj))
            sampled = label_subj.sample(n=n, random_state=random_seed + k)
            few_shot_parts.append(sampled)
        few_shot_subjects = pd.concat(few_shot_parts)
        few_shot_data = get_data(few_shot_subjects)

        few_shot_data.to_csv(os.path.join(output_dir, f'{split_name}_train_k{k}.csv'), index=False)
        counts = Counter(few_shot_data['label'])
        log(f"Few-shot {k}/class: {len(few_shot_data)} vols, Class dist: {dict(counts)}")

# === LOAD UCSF DATA ===
data = pd.read_csv(input_path)

# === SPLIT 1: Grade 2 vs Grade 3+4 (label 2 vs 3/4)
df_2_vs_34 = data[data['label'].isin([2, 3, 4])].copy()
label_map_2_vs_34 = {2: 0, 3: 1, 4: 1}
subject_wise_balanced_split(df_2_vs_34, label_pair=[2, 3, 4], label_map=label_map_2_vs_34, split_name='ucsf_grade2_vs_34')

# === SPLIT 2: Grade 2+3 vs Grade 4 (label 2/3 vs 4)
df_23_vs_4 = data[data['label'].isin([2, 3, 4])].copy()
label_map_23_vs_4 = {2: 0, 3: 0, 4: 1}
subject_wise_balanced_split(df_23_vs_4, label_pair=[2, 3, 4], label_map=label_map_23_vs_4, split_name='ucsf_grade23_vs_4')

# === SAVE LOG ===
with open(log_path, 'w') as f:
    f.write("\n".join(log_lines))
log("All UCSF balanced splits and logs saved.")
