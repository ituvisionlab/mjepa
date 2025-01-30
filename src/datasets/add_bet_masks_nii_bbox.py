import os
import csv
import subprocess
import nibabel as nib
import numpy as np
import pandas as pd

# Input and output CSV files
input_csv = "/gpfs/home/unalg01/jepa/src/datasets/ppmi_all_nii.csv"
output_csv = "/gpfs/home/unalg01/jepa/src/datasets/ppmi_all_nii_with_bbox.csv"

# Function to calculate the bounding box of the brain mask
def calculate_bbox(mask_file):
    try:
        # Load the brain mask as a NIfTI file
        mask_nii = nib.load(mask_file)
        mask_data = mask_nii.get_fdata()

        # Find the non-zero coordinates
        coords = np.argwhere(mask_data > 0)

        # Calculate the bounding box (min and max along each axis)
        if coords.size == 0:  # If the mask is empty
            return None
        bbox = {
            "xmin": coords[:, 0].min(),
            "xmax": coords[:, 0].max(),
            "ymin": coords[:, 1].min(),
            "ymax": coords[:, 1].max(),
            "zmin": coords[:, 2].min(),
            "zmax": coords[:, 2].max(),
        }
        return bbox
    except Exception as e:
        print(f"Error calculating bounding box for {mask_file}: {e}")
        return None

# Function to perform brain extraction using FSL's bet command
def run_bet(input_file, output_file):
    try:
        # Run the FSL bet command
        bet_cmd = ["bet", input_file, output_file, "-m"]
        subprocess.run(bet_cmd, check=True)
        print(f"Brain extraction completed: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running bet for {input_file}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Read the CSV file
data = pd.read_csv(input_csv)

# New columns for bounding box fields
bbox_columns = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
for col in bbox_columns:
    data[col] = None

# Process each NIfTI file in the CSV
for idx, row in data.iterrows():
    nii_file_path = row["nii_file_path"]

    # Ensure the file exists
    if not os.path.exists(nii_file_path):
        print(f"File not found: {nii_file_path}")
        continue

    # Create the output brain mask file path
    nii_dir, nii_filename = os.path.split(nii_file_path)
    base_name, ext = os.path.splitext(nii_filename)
    if ext == ".gz":  # Handle .nii.gz files
        base_name = os.path.splitext(base_name)[0]
    bet_output_path = os.path.join(nii_dir, f"{base_name}_betmask.nii.gz")

    # Perform brain extraction
    run_bet(nii_file_path, bet_output_path)

    # Calculate the bounding box
    bbox = calculate_bbox(bet_output_path)

    # Add bounding box information to the row
    if bbox:
        for col in bbox_columns:
            data.at[idx, col] = bbox[col]
    else:
        print(f"No brain mask found for {nii_file_path}")

# Save the updated data to a new CSV file
data.to_csv(output_csv, index=False)
print(f"Updated CSV saved with bounding box fields: {output_csv}")
