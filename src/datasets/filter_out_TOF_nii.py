import pandas as pd

# Input and Output CSV file paths
input_csv = "/gpfs/home/unalg01/jepa/src/datasets/oasis3_all_bet_nii_with_bbox.csv"  
output_csv = "/gpfs/home/unalg01/jepa/src/datasets/oasis3_all_bet.csv"  # New filtered CSV file

# Load CSV file
df = pd.read_csv(input_csv)

# Filter out TOF files
df_filtered = df[df["contrast"] != "TOF"]

# Save the filtered data to a new CSV file
df_filtered.to_csv(output_csv, index=False)

print(f"Filtered CSV saved: {output_csv}")
print(f"Original dataset size: {len(df)}")
print(f"Filtered dataset size: {len(df_filtered)}")
