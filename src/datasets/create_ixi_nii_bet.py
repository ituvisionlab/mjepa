import os
import csv
import subprocess
import nibabel as nib
import numpy as np
import glob

# Paths
base_data_path = "/gpfs/data/sodicksonlab/gozde/IXI"
output_csv = "/gpfs/data/sodicksonlab/gozde/IXI/ixi_all_nii_with_bbox.csv"

# Default values for missing fields
default_date = "2000-01-01"
default_sex = "X"
default_age = "0.00"
default_weight = "0.00"

# CSV Header
csv_header = ["label", "subject_id", "contrast", "date_acquired", "subject_sex", 
              "subject_age", "subject_weight", "nii_file_path", "xmin", "xmax", 
              "ymin", "ymax", "zmin", "zmax"]

# Function to extract subject_id from file name
def extract_subject_id(filename):
    parts = filename.split("-")
    if len(parts) >= 2 and parts[0].startswith("IXI"):
        return parts[0][3:]  # Extract the numeric subject ID (e.g., "159" from "IXI159")
    return "Unknown"

# Function to perform brain extraction using BET
def run_bet(input_file, output_file):
    try:
        # Run BET command
        bet_cmd = ["bet", input_file, output_file, "-m"]
        subprocess.run(bet_cmd, check=True)
        print(f"Brain extraction completed: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running BET for {input_file}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Function to calculate the bounding box of the brain mask
def calculate_bbox(mask_file):
    try:
        mask_nii = nib.load(mask_file)
        mask_data = mask_nii.get_fdata()

        # Find non-zero coordinates
        coords = np.argwhere(mask_data > 0)

        # Calculate bounding box
        if coords.size == 0:  
            return None
        bbox = {
            "xmin": int(coords[:, 0].min()),
            "xmax": int(coords[:, 0].max()),
            "ymin": int(coords[:, 1].min()),
            "ymax": int(coords[:, 1].max()),
            "zmin": int(coords[:, 2].min()),
            "zmax": int(coords[:, 2].max()),
        }
        return bbox
    except Exception as e:
        print(f"Error calculating bounding box for {mask_file}: {e}")
        return None

# Scan for NIfTI files in IXI dataset
nii_files = glob.glob(f"{base_data_path}/**/*.nii.gz", recursive=True)

# Initialize CSV rows
csv_rows = []
processed_count = 0

for nii_file_path in sorted(nii_files):
    if "-seg.nii.gz" in nii_file_path:
        continue  # Skip segmentation files

    processed_count += 1
    print(f"Processing file {processed_count}: {nii_file_path}")

    # Extract subject ID
    nii_filename = os.path.basename(nii_file_path)
    subject_id = extract_subject_id(nii_filename)
    contrast = "T1"  # All images are T1

    # Construct expected brain mask file paths
    nii_dir = os.path.dirname(nii_file_path)
    base_name = nii_filename.replace(".nii.gz", "")
    bet_output_path = os.path.join(nii_dir, f"{base_name}_betmask.nii.gz")
    bet_mask_path = os.path.join(nii_dir, f"{base_name}_betmask_mask.nii.gz")

    # Check if brain extraction is needed
    if not (os.path.exists(bet_output_path) and os.path.exists(bet_mask_path)):
        run_bet(nii_file_path, bet_output_path)

    # Compute bounding box
    bbox = calculate_bbox(bet_mask_path)

    # If bounding box computation failed, use full volume size
    if bbox is None:
        img = nib.load(nii_file_path)
        xsize, ysize, zsize = img.shape[:3]
        bbox = {"xmin": 0, "xmax": xsize, "ymin": 0, "ymax": ysize, "zmin": 0, "zmax": zsize}
        print(f"No brain mask found for {nii_file_path}, using full volume as bounding box.")

    # Append data to CSV rows
    csv_rows.append([0, subject_id, contrast, default_date, default_sex, default_age, 
                     default_weight, nii_file_path, bbox["xmin"], bbox["xmax"], 
                     bbox["ymin"], bbox["ymax"], bbox["zmin"], bbox["zmax"]])

# Write the results to a new CSV file
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)
    writer.writerows(csv_rows)

print(f"IXI dataset CSV saved with bounding boxes: {output_csv}")
