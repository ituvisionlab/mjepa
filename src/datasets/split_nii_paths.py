import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
#file_path = '/gpfs/home/unalg01/jepa/src/datasets/mnist3d/nii_volumes.csv'  # Replace with your CSV file path
file_path = '/gpfs/home/unalg01/jepa/src/datasets/updated_nii_file_paths.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Specify the split ratio
train_ratio = 0.8  # 80% training
val_ratio = 0.2    # 20% validation

# Split the data
train_data, val_data = train_test_split(data, test_size=val_ratio, random_state=42)

# Save the splits to new CSV files
train_data.to_csv('/gpfs/home/unalg01/jepa/src/datasets/adni_train_split.csv', index=False)
val_data.to_csv('/gpfs/home/unalg01/jepa/src/datasets/adni_val_split.csv', index=False)

print("Data has been split and saved to 'adni_train_split.csv' and 'adni_val_split.csv'.")
