import pandas as pd

# Load the full input CSV
file_path = '/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_nii_verified.csv'
df = pd.read_csv(file_path)

# Count volumes per label
label_counts = df['label'].value_counts().sort_index()

# Compute totals
label_0_1_2 = label_counts[[0, 1, 2]].sum()
label_rest = label_counts[label_counts.index > 2].sum()
total = label_counts.sum()

# Print breakdown
print("Volume counts per label:")
print(label_counts)
print("\nSummary:")
print(f"Total volumes: {total}")
print(f"Total for labels 0,1,2: {label_0_1_2}")
print(f"Total for labels >2 (excluded): {label_rest}")
