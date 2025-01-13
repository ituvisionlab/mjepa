import os
import csv
import json
import re
import nibabel as nib

# Paths to OASIS3 dataset and metadata CSV
base_data_path = "/gpfs/data/sodicksonlab/gozde/OASIS3all"
output_csv = "oasis3_all_nii.csv"

# Excluded subfolder keywords and contrasts
excluded_subfolders = ["fmap", "func", "swi", "dwi"]
excluded_contrasts = [
    "mIP", "bold", "SWI", "unknown", "DTI", "ASL", "dwi", "field_mapping",
    "bold-rest", "epd2d", "MDDW", "rsfmri", "Mag_", "Pha_"
]

# Mapping of specific contrasts
contrast_mapping = {
    "T1": "T1",
    "T2": "T2",
    "MPRAGE": "MPRAGE",
    "FLAIR": "FLAIR",
    "T2_star": "T2star",
    "TOF": "TOF",
    "tse": "tse"
}

# Initialize CSV header and rows
csv_header = ["label", "subject_id", "contrast", "nii_file_path"]
csv_rows = []

# Function to extract subject ID from the filename
def extract_subject_id(filename):
    match = re.search(r"sub-(OAS\d+)", filename)
    if match:
        return match.group(1)
    return "Unknown"

# Function to determine if a folder or file should be excluded
def is_excluded(path, filename):
    if any(excl in path for excl in excluded_subfolders):
        return True
    if any(excl in filename for excl in excluded_contrasts):
        return True
    return False

# Function to map contrast based on description and filename
def map_contrast(series_description, filename):
    for key, mapped_value in contrast_mapping.items():
        if key.lower() in series_description.lower() or key.lower() in filename.lower():
            return mapped_value
    return "Unknown"

# Function to filter NIfTI files based on FOV and spacing
def filter_nifti(img, min_fov=50, max_spacing=6.5):
    header = img.header
    voxel_spacing = header.get_zooms()[:3]
    image_dimensions = header.get_data_shape()[:3]
    fov = [spacing * dim for spacing, dim in zip(voxel_spacing, image_dimensions)]
    if any(f < min_fov for f in fov) or any(s > max_spacing for s in voxel_spacing):
        return False
    return True

cntr = 0

# Walk through the dataset folders
for root, dirs, files in os.walk(base_data_path):
    if any(excl in root for excl in excluded_subfolders):
        continue  # Skip excluded subfolders
    
    for file in files:
        if file.endswith(".nii.gz"):
            nii_file_path = os.path.join(root, file)
            cntr += 1
            print(f"Processing file {cntr}: {file}")

            # Skip excluded files
            if is_excluded(root, file):
                print(f"Excluded file: {nii_file_path}")
                continue

            # Skip small files
            if os.path.getsize(nii_file_path) < 2e6:  # Skip files smaller than 2 MB
                print(f"Skipped small file: {nii_file_path}")
                continue
            
            # Load the NIfTI image and apply FOV and spacing filter
            try:
                img = nib.load(nii_file_path)
                if not filter_nifti(img):
                    print(f"Excluded based on FOV/spacing: {nii_file_path}")
                    continue
            except Exception as e:
                print(f"Error loading NIfTI file {nii_file_path}: {e}")
                continue

            # Extract subject ID from the filename
            subject_id = extract_subject_id(file)

            # Find the corresponding JSON file
            json_file_name = file.replace(".nii.gz", ".json")
            json_file_path = os.path.join(root, json_file_name)
            
            if not os.path.exists(json_file_path):
                print(f"Warning: JSON file not found for {nii_file_path}")
                continue

            # Read metadata from the JSON file
            try:
                with open(json_file_path, "r") as json_file:
                    metadata = json.load(json_file)
                    series_description = metadata.get("SeriesDescription", "Unknown")
            except json.JSONDecodeError:
                print(f"Warning: Unable to parse {json_file_path}")
                continue
            
            contrast = map_contrast(series_description, file)
            
            # Append the valid row to the CSV
            csv_rows.append([0, subject_id, contrast, nii_file_path])

# Write the rows to the output CSV
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)
    writer.writerows(csv_rows)

# Print status
print(f"CSV file created: {output_csv}")
print(f"Total NIfTI files processed: {len(csv_rows)}")
