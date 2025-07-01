import pandas as pd

# Input and output paths
input_csv = "/gpfs/home/unalg01/jepa/src/datasets/SCAN_NIFTI_all_final.csv"
output_csv = "/gpfs/home/unalg01/jepa/src/datasets/SCAN_NIFTI_all_filtered.csv"

# Load CSV
df = pd.read_csv(input_csv)

# Filter out rows where '_ME_' is in the file path
filtered_df = df[~df['nii_file_path'].str.contains('_ME_')]
# Filter out entries with "_3TE_" in the path (case-insensitive)
filtered_df = filtered_df[~filtered_df["nii_file_path"].str.contains("_3TE_", case=False, na=False)]

# Save the result
filtered_df.to_csv(output_csv, index=False)

print(f"[DONE] Saved filtered CSV to: {output_csv}")
print(f"[INFO] Original rows: {len(df)}, Filtered rows: {len(filtered_df)}")
