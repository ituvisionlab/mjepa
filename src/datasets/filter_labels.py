import pandas as pd

# Input CSV file
#input_csv = "/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_train.csv"
input_csv = "/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_val.csv"

# Output CSV file (filtered version)
#output_csv = "/gpfs/home/unalg01/jepa/src/datasets/adni_all3class_bet_train.csv"
output_csv = "/gpfs/home/unalg01/jepa/src/datasets/adni_all_3class_bet_val.csv"

# Load the CSV file
df = pd.read_csv(input_csv)

# Keep only rows where label is 0, 1, or 2
df_filtered = df[df["label"].isin([0, 1, 2])]

# Save the filtered data to a new CSV file
df_filtered.to_csv(output_csv, index=False)

print(f"Filtered CSV saved with labels 0, 1, 2: {output_csv}")
print(f"Original dataset size: {len(df)}")
print(f"Filtered dataset size: {len(df_filtered)}")
