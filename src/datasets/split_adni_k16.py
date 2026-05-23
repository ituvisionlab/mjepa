import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
input_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_cv_folds_stratified/adni_downstream.csv'
output_dir = '/gpfs/home/unalg01/jepa/src/datasets/adni_downstream_splits'
random_seed = 1938 #1881
np.random.seed(random_seed)

# === LOAD DATA ===
data = pd.read_csv(input_path)

# === FILTER FOR NC vs AD ONLY ===
label_pair = [0, 2]
label_mapping = {label_pair[0]: 0, label_pair[1]: 1}
df_filtered = data[data['label'].isin(label_pair)].copy()
df_filtered['label'] = df_filtered['label'].map(label_mapping)

# Subject-wise labels
subject_labels = df_filtered[['subject_id', 'label']].drop_duplicates('subject_id')

# Split as before
trainval_subjects, _ = train_test_split(
    subject_labels,
    test_size=0.2,
    stratify=subject_labels['label'],
    random_state=random_seed
)

train_subjects, _ = train_test_split(
    trainval_subjects,
    test_size=0.2,
    stratify=trainval_subjects['label'],
    random_state=random_seed
)

# === RECREATE K=16 FEW-SHOT SPLIT ===
k = 16
few_shot_parts = []
for label in [0, 1]:
    label_subjects = train_subjects[train_subjects['label'] == label]
    n = min(k, len(label_subjects))
    sampled_subjects = label_subjects.sample(n=n, random_state=random_seed + k)
    few_shot_parts.append(sampled_subjects)

few_shot_subjects = pd.concat(few_shot_parts)
few_shot_data = df_filtered[df_filtered['subject_id'].isin(few_shot_subjects['subject_id'])]

# === SAVE ===
few_shot_path = os.path.join(output_dir, 'nc_ad_train_k16.csv')
few_shot_data.to_csv(few_shot_path, index=False)

print("Recreated nc_ad_train_k16.csv")
