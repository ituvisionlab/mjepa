import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Load the original dataset
original_csv = "/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_val.csv"
df = pd.read_csv(original_csv)

# Ensure subject_id exists
if 'subject_id' not in df.columns:
    raise ValueError("Column 'subject_id' not found in the CSV.")

# Split function avoiding subject leakage
def split_by_subject(dataframe, test_size=0.2, random_state=42):
    unique_subjects = dataframe['subject_id'].unique()
    train_subjects, val_subjects = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state)
    
    train_df = dataframe[dataframe['subject_id'].isin(train_subjects)]
    val_df = dataframe[dataframe['subject_id'].isin(val_subjects)]
    
    return train_df, val_df

# Print label distributions
def print_label_distribution(name, df_train, df_val):
    print(f"\n{name} Class Distribution:")
    print("Train Set:")
    print(df_train['label'].value_counts())
    print(df_train['label'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    print("Validation Set:")
    print(df_val['label'].value_counts())
    print(df_val['label'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

### Scenario 1: 2-Class (0 = NC, 1 = AD)
df_2class_ad = df[df["label"].isin([0, 2])].copy()
df_2class_ad["label"] = df_2class_ad["label"].replace({2: 1})  # Relabel AD (2) to 1

train_2class_ad, val_2class_ad = split_by_subject(df_2class_ad)
train_2class_ad.to_csv("/gpfs/home/unalg01/jepa/src/datasets/adni_all_2class_bet_valtrain.csv", index=False)
val_2class_ad.to_csv("/gpfs/home/unalg01/jepa/src/datasets/adni_all_2class_bet_valtest.csv", index=False)

print_label_distribution("[2-Class NC vs AD]", train_2class_ad, val_2class_ad)

# Oversample AD in Scenario 1 train set
df_nc = train_2class_ad[train_2class_ad["label"] == 0]
df_ad = train_2class_ad[train_2class_ad["label"] == 1]

target_count = len(df_nc)

df_ad_oversampled = resample(
    df_ad,
    replace=True,
    n_samples=target_count,
    random_state=42
)

df_balanced_2class = pd.concat([df_nc, df_ad_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced_2class.to_csv("/gpfs/home/unalg01/jepa/src/datasets/adni_all_2class_bet_valtrain_oversampled.csv", index=False)

print("\n[2-Class NC vs AD with AD Oversampled] New Train Set Distribution:")
print(df_balanced_2class['label'].value_counts())
print(df_balanced_2class['label'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

### Scenario 2: 3-Class (0 = NC, 1 = MCI, 2 = AD)
df_3class = df[df["label"].isin([0, 1, 2])].copy()
train_3class, val_3class = split_by_subject(df_3class)

train_3class.to_csv("/gpfs/home/unalg01/jepa/src/datasets/adni_all_3class_bet_valtrain.csv", index=False)
val_3class.to_csv("/gpfs/home/unalg01/jepa/src/datasets/adni_all_3class_bet_valtest.csv", index=False)

print_label_distribution("[3-Class NC vs MCI vs AD]", train_3class, val_3class)

# Oversample AD in Scenario 2 train set
df_nc = train_3class[train_3class["label"] == 0]
df_mci = train_3class[train_3class["label"] == 1]
df_ad = train_3class[train_3class["label"] == 2]

target_count = len(df_nc)

df_ad_oversampled = resample(
    df_ad,
    replace=True,
    n_samples=target_count,
    random_state=42
)

df_balanced_3class = pd.concat([df_nc, df_mci, df_ad_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced_3class.to_csv("/gpfs/home/unalg01/jepa/src/datasets/adni_all_3class_bet_valtrain_oversampled.csv", index=False)

print("\n[3-Class with AD Oversampled] New Train Set Distribution:")
print(df_balanced_3class['label'].value_counts())
print(df_balanced_3class['label'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

### Scenario 3: 2-Class (0 = NC, 1 = MCI)
df_2class_mci = df[df["label"].isin([0, 1])].copy()
train_2class_mci, val_2class_mci = split_by_subject(df_2class_mci)

train_2class_mci.to_csv("/gpfs/home/unalg01/jepa/src/datasets/adni_all_2class_nc_mci_bet_valtrain.csv", index=False)
val_2class_mci.to_csv("/gpfs/home/unalg01/jepa/src/datasets/adni_all_2class_nc_mci_bet_valtest.csv", index=False)

print_label_distribution("[2-Class NC vs MCI]", train_2class_mci, val_2class_mci)
