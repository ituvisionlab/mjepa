import os
import pandas as pd

# Input and output CSV file paths
input_csv = "/gpfs/home/unalg01/jepa/src/datasets/adni_all_nii_with_bbox.csv"  # Original CSV file
output_csv = "/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_nii_with_bbox.csv"  # New CSV file

# Load the CSV file
data = pd.read_csv(input_csv)

# Modify the nii_file_path to use the _betmask.nii.gz version
def modify_nii_path(nii_path):
    """Modify the NIfTI file path to use the _betmask.nii.gz version."""
    nii_dir, nii_filename = os.path.split(nii_path)  # Extract directory and filename
    base_name, ext = os.path.splitext(nii_filename)  # Remove extension
    if ext == ".gz":  # Handle .nii.gz case
        base_name, _ = os.path.splitext(base_name)

    # Construct the new filename
    new_filename = f"{base_name}_betmask.nii.gz"
    new_path = os.path.join(nii_dir, new_filename)
    return new_path

# Apply the modification to the nii_file_path column
data["nii_file_path"] = data["nii_file_path"].apply(modify_nii_path)

# Save the updated data to a new CSV file
data.to_csv(output_csv, index=False)

print(f"Updated CSV saved as {output_csv}")
