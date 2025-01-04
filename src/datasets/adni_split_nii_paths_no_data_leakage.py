import pandas as pd
from sklearn.model_selection import train_test_split

# Load the train and validation CSV files
train_file_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_train_split.csv'
val_file_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_val_split.csv'

train_data = pd.read_csv(train_file_path)
val_data = pd.read_csv(val_file_path)

# Combine the two datasets
data = pd.concat([train_data, val_data], ignore_index=True)

# Ensure there is a 'subject_id' column in the data
group_col = 'subject_id'  # Correct column name for subject IDs
if group_col not in data.columns:
    raise ValueError(f"The column '{group_col}' does not exist in the provided CSV files.")

# Get unique subject IDs
unique_subjects = data[group_col].unique()

# Specify the split ratio
train_ratio = 0.8  # 80% training
val_ratio = 0.2    # 20% validation

# Split the subjects into train and validation sets
train_subjects, val_subjects = train_test_split(unique_subjects, test_size=val_ratio, random_state=42)

# Filter the original data based on the split subjects
new_train_data = data[data[group_col].isin(train_subjects)]
new_val_data = data[data[group_col].isin(val_subjects)]

# Save the new splits to CSV files
new_train_file_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_new_train_split.csv'
new_val_file_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_new_val_split.csv'

new_train_data.to_csv(new_train_file_path, index=False)
new_val_data.to_csv(new_val_file_path, index=False)

print("Data has been re-split by subject_id and saved to 'adni_new_train_split.csv' and 'adni_new_val_split.csv'.")
