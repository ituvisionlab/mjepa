import os
import csv
import glob
import nibabel as nib
import numpy as np

# Paths
base_data_path = "/gpfs/data/sodicksonlab/gozde/MOOD/brain_train"
output_csv = "/gpfs/data/sodicksonlab/gozde/MOOD/brain_train/mood_all_nii_with_bbox.csv"

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
    if len(parts) >= 2:
        return parts[-1].split(".")[0]  # Extract ID from last section before .nii.gz
    return "Unknown"

# Function to calculate the bounding box of the brain volume
def calculate_bbox(nifti_file):
    try:
        img = nib.load(nifti_file)
        volume_data = img.get_fdata()

        # Find non-zero coordinates
        coords = np.argwhere(volume_data > 0)

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
        print(f"Error calculating bounding box for {nifti_file}: {e}")
        return None

# Scan for NIfTI files in MOOD dataset
nii_files = glob.glob(f"{base_data_path}/**/*.nii.gz", recursive=True)

# Initialize CSV rows
csv_rows = []
processed_count = 0

for nii_file_path in sorted(nii_files):
    processed_count += 1
    print(f"Processing file {processed_count}: {nii_file_path}")

    # Extract subject ID
    nii_filename = os.path.basename(nii_file_path)
    subject_id = extract_subject_id(nii_filename)
    contrast = "T1"  # Assuming all images are T1-weighted

    # Compute bounding box directly from image
    bbox = calculate_bbox(nii_file_path)

    # If bounding box computation failed, use full volume size
    if bbox is None:
        img = nib.load(nii_file_path)
        xsize, ysize, zsize = img.shape[:3]
        bbox = {"xmin": 0, "xmax": xsize, "ymin": 0, "ymax": ysize, "zmin": 0, "zmax": zsize}
        print(f"Warning: No non-zero voxels found in {nii_file_path}, using full volume as bounding box.")

    # Append data to CSV rows
    csv_rows.append([0, subject_id, contrast, default_date, default_sex, default_age, 
                     default_weight, nii_file_path, bbox["xmin"], bbox["xmax"], 
                     bbox["ymin"], bbox["ymax"], bbox["zmin"], bbox["zmax"]])

# Write the results to a new CSV file
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)
    writer.writerows(csv_rows)

print(f"MOOD dataset CSV saved with bounding boxes: {output_csv}")
