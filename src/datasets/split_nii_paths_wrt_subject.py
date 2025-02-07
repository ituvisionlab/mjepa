import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_nii_with_bbox.csv' 
# file_path = '/gpfs/home/unalg01/jepa/src/datasets/ppmi_all_nii_with_bbox.csv' 
# file_path = '/gpfs/home/unalg01/jepa/src/datasets/oasis3_all_nii.csv' 
data = pd.read_csv(file_path)

# Specify the split ratio
train_ratio = 0.80  # % training
val_ratio = 0.20    # % validation

# Ensure there is a 'subject_id' column in the data
group_col = 'subject_id'  # Correct column name for subject IDs
if group_col not in data.columns:
    raise ValueError(f"The column '{group_col}' does not exist in the provided CSV file.")

# Get unique subject IDs
unique_subjects = data[group_col].unique()

# Split the subjects into train and validation sets
train_subjects, val_subjects = train_test_split(unique_subjects, test_size=val_ratio, random_state=42)

# Filter the original data based on the split subjects
train_data = data[data[group_col].isin(train_subjects)]
val_data = data[data[group_col].isin(val_subjects)]

# Save the splits to new CSV files
train_data.to_csv('/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_train.csv', index=False)
val_data.to_csv('/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_val.csv', index=False)

print("Data has been split by subject_id and saved to 'train_split.csv' and 'val_split.csv'.")
