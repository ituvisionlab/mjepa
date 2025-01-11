import os
import csv
import json

# Paths to OASIS3 dataset and metadata CSV
base_data_path = "/gpfs/data/sodicksonlab/gozde/OASIS3all"
output_csv = "oasis3_all_nii.csv"

# Excluded contrasts and keywords in file names
excluded_contrasts = [
    "mIP", "bold", "SWI", "unknown", "DTI", "ASL", "dwi", "field_mapping",
    "bold-rest", "epd2d", "MDDW", "rsfmri", "Mag_", "Pha_"
]

excluded_keywords = ["fieldmap", "swi"]

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

# Function to extract metadata from the JSON file
def read_json_metadata(json_file_path):
    try:
        with open(json_file_path, "r") as json_file:
            metadata = json.load(json_file)
            subject_id = metadata.get("subject_id", "Unknown")
            series_description = metadata.get("SeriesDescription", "Unknown")
            return subject_id, series_description
    except json.JSONDecodeError:
        print(f"Warning: Unable to parse {json_file_path}")
        return None, None

# Function to determine if a file should be excluded based on keywords
def is_excluded(filename):
    return any(excluded in filename for excluded in excluded_contrasts + excluded_keywords)

# Function to map contrast based on description and filename
def map_contrast(series_description, filename):
    for key, mapped_value in contrast_mapping.items():
        if key.lower() in series_description.lower() or key.lower() in filename.lower():
            return mapped_value
    return "Unknown"

cntr = 0

# Walk through the dataset folders
for root, dirs, files in os.walk(base_data_path):
    for file in files:
        if file.endswith(".nii.gz"):
            nii_file_path = os.path.join(root, file)
            cntr += 1
            print(f"Processing file {cntr}: {file}")

            # Extract subject_id from the folder structure
            folder_parts = root.split(os.sep)
            subject_id = folder_parts[-2]  # e.g., OAS31048_MR_d3195

            # Skip excluded files
            if is_excluded(file):
                print(f"Excluded file: {nii_file_path}")
                continue
            
            # Find the corresponding JSON file
            json_file_name = file.replace(".nii.gz", ".json")
            json_file_path = os.path.join(root, json_file_name)
            
            if not os.path.exists(json_file_path):
                print(f"Warning: JSON file not found for {nii_file_path}")
                continue

            # Read metadata from the JSON file
            subject_id, series_description = read_json_metadata(json_file_path)
            if subject_id is None or series_description is None:
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
