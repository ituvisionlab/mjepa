import os
import csv
import nibabel as nib
import numpy as np
import re

# Paths to BraTS2024 dataset subdirectories
base_data_paths = [
    "/gpfs/data/sodicksonlab/gozde/BraTS2024/training_data1_v2",
    "/gpfs/data/sodicksonlab/gozde/BraTS2024/training_data_additional"
]

# Output CSV file
output_csv = "brats24_all_nii.csv"

# Placeholder values for missing fields
default_sex = "X"
default_age = "0.00"
default_weight = "0.00"

# Mapping of specific contrasts
contrast_mapping = {
    "t1c": "T1",
    "t1n": "T1",
    "t2w": "T2",
    "t2f": "T2"
}

# CSV Header
csv_header = [
    "label", "subject_id", "contrast", "date_acquired",
    "subject_sex", "subject_age", "subject_weight", "nii_file_path",
    "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"
]

# Function to map contrast from filename
def map_contrast(filename):
    for key, mapped_value in contrast_mapping.items():
        if key in filename.lower():
            return mapped_value
    return "Unknown"

# Function to extract numeric portion as 'date_acquired'
def extract_date_acquired(filename):
    match = re.search(r"-(\d{3})", filename)  # Extracts last three-digit number
    return match.group(1) if match else "Unknown"

# Function to calculate bounding box of nonzero intensities
def calculate_bbox(nii_file):
    try:
        nii = nib.load(nii_file)
        volume = nii.get_fdata()
        nonzero_coords = np.argwhere(volume > 0)

        if nonzero_coords.size == 0:
            return None

        bbox = {
            "xmin": int(nonzero_coords[:, 0].min()),
            "xmax": int(nonzero_coords[:, 0].max()),
            "ymin": int(nonzero_coords[:, 1].min()),
            "ymax": int(nonzero_coords[:, 1].max()),
            "zmin": int(nonzero_coords[:, 2].min()),
            "zmax": int(nonzero_coords[:, 2].max())
        }
        return bbox
    except Exception as e:
        print(f"Error calculating bounding box for {nii_file}: {e}")
        return None

# Collect all rows for the CSV
csv_rows = []
unique_subject_ids = set()
processed_count = 0

# Process both dataset folders
for base_data_path in base_data_paths:
    for root, _, files in os.walk(base_data_path):
        for file in files:
            if file.endswith(".nii.gz") and not file.endswith("-seg.nii.gz"):  # Exclude segmentation files
                nii_file_path = os.path.join(root, file)
                processed_count += 1
                print(f"Processing file {processed_count}: {file}")

                # Extract subject ID (first part of filename)
                subject_id = file.split('-')[2]  # Extracts "02597" from "BraTS-GLI-02597-100"

                # Extract date acquired from filename
                date_acquired = extract_date_acquired(file)

                # Determine contrast
                contrast = map_contrast(file)

                # Calculate bounding box
                bbox = calculate_bbox(nii_file_path)
                if bbox is None:
                    print(f"No brain mask found for {nii_file_path}, skipping.")
                    continue

                # Append data row
                csv_rows.append([
                    0, subject_id, contrast, date_acquired,  # label=0 (placeholder)
                    default_sex, default_age, default_weight, nii_file_path,
                    bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"], bbox["zmin"], bbox["zmax"]
                ])
                unique_subject_ids.add(subject_id)

# Write data to CSV file
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)
    writer.writerows(csv_rows)

# Write summary log file
log_file_path = "brats24_dataset_summary.log"
with open(log_file_path, mode="w") as log_file:
    log_file.write(f"Total unique subjects: {len(unique_subject_ids)}\n")
    log_file.write(f"Total processed NIfTI files: {processed_count}\n")
    log_file.write(f"CSV file created: {output_csv}\n")

print(f"Dataset processing complete. Summary saved to {log_file_path}.")
