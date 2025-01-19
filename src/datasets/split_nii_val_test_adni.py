import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_all_val_split.csv' 
data = pd.read_csv(file_path)

# Specify the split ratio
train_ratio = 0.5  # 50% training
val_ratio = 0.5    # 50% validation

# Ensure required columns exist in the data
group_col = 'subject_id'  # Column for subject IDs
label_col = 'label'       # Column for labels
if group_col not in data.columns or label_col not in data.columns:
    raise ValueError(f"Columns '{group_col}' and/or '{label_col}' do not exist in the provided CSV file.")

# Filter the data to only include labels 0, 1, and 2
filtered_data = data[data[label_col].isin([0, 1, 2])]

# Get unique subject IDs and their corresponding labels
subject_labels = filtered_data.groupby(group_col)[label_col].first().reset_index()

# Perform stratified split based on labels
train_subjects, val_subjects = train_test_split(
    subject_labels[group_col],
    test_size=val_ratio,
    stratify=subject_labels[label_col],
    random_state=42
)

# Filter the original data based on the split subjects
train_data = filtered_data[filtered_data[group_col].isin(train_subjects)]
val_data = filtered_data[filtered_data[group_col].isin(val_subjects)]

# Save the splits to new CSV files
train_data.to_csv('/gpfs/home/unalg01/jepa/src/datasets/adni_all_valtrain_split.csv', index=False)
val_data.to_csv('/gpfs/home/unalg01/jepa/src/datasets/adni_all_valtest_split.csv', index=False)

print("Data has been split with stratification and saved to 'valtrain_split.csv' and 'valtest_split.csv'.")
