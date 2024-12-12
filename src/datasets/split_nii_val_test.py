import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_val_split.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Specify the split ratio
val_ratio = 0.8  # 80% 
test_ratio = 0.2    # 20% test

# Split the data
train_data, val_data = train_test_split(data, test_size=test_ratio, random_state=42)

# Save the splits to new CSV files
train_data.to_csv('/gpfs/home/unalg01/jepa/src/datasets/adni_valtrain_split.csv', index=False)
val_data.to_csv('/gpfs/home/unalg01/jepa/src/datasets/adni_valtest_split.csv', index=False)

print("Data has been split and saved to 'adni_valtrain_split.csv' and 'andi_valtest_split.csv'.")
