import os
import csv

import nibabel as nib

# Paths to PPMI dataset and metadata
base_data_path = "/gpfs/data/sodicksonlab/gozde/ADNI"
meta_data_path = "/gpfs/data/sodicksonlab/gozde/ADNI-meta/ADNI"

# Output CSV file
output_csv = "adni_all_nii.csv"

label_mapping = {
    "NC": 0,
    "MCI": 1,
    "AD": 2,
    "Other": 3
}

# Function to extract label from XML using string search
def extract_label_from_xml(xml_path):
    try:
        with open(xml_path, "r") as file:
            content = file.read()
            # Search for the <researchGroup> tag
            start_tag = "<researchGroup>"
            end_tag = "</researchGroup>"
            start_idx = content.find(start_tag)
            end_idx = content.find(end_tag)
            
            if start_idx != -1 and end_idx != -1:
                label_text = content[start_idx + len(start_tag):end_idx].strip()
                if label_text not in label_mapping:
                    # Assign a new integer label to unseen labels
                    label_mapping[label_text] = len(label_mapping)
                return label_mapping[label_text]
            else:
                print(f"Warning: <researchGroup> tag not found in XML: {xml_path}")
                return None
    except Exception as e:
        print(f"Error reading XML {xml_path}: {e}")
        return None

def filter_nifti(img, min_fov=50, max_spacing=6.5):
       
    header = img.header
    
    # Extract voxel dimensions and image dimensions
    voxel_spacing = header.get_zooms()[:3]  # pixdim[1:3]
    image_dimensions = header.get_data_shape()[:3]  # dim[1:3]
    
    # Compute field of view for each axis
    fov = [spacing * dim for spacing, dim in zip(voxel_spacing, image_dimensions)]
    
    # Apply filtering criteria
    if any(f < min_fov for f in fov) or any(s > max_spacing for s in voxel_spacing):
        return False  # Exclude this file
    return True  # Include this file
    
# Collect all rows for the CSV
csv_data = []

# Walk through the dataset folders
for root, dirs, files in os.walk(base_data_path):
    for file in files:
        if file.endswith(".nii"):
            # Construct the absolute path to the .nii file
            nii_file_path = os.path.join(root, file)
            
            if os.path.getsize(nii_file_path) < 2e6:
                continue
            
            img = nib.load(nii_file_path)
            include_this_file = filter_nifti(img) #filter wrt fov>50 & spacing >6.5mm
            
            if not include_this_file:
                continue
            
            # Extract subject ID, folder name, study ID, and image ID from the .nii file path
            path_parts = nii_file_path.split(os.sep)
            subject_id = path_parts[-5]  # e.g., "51632"
            folder_name = path_parts[-4]  # e.g., "T1-anatomical"
            study_date = path_parts[-3]  # e.g., "2014-02-26_10_55_06.0"
            image_id = path_parts[-2].lstrip("I")  # e.g., "696651"
            
            # Extract study ID from the .nii file name
            study_id = file.split('_S')[-1].split('_I')[0]  # Extract study ID (e.g., "331865")
            
            # Construct the corresponding XML file path
            xml_file_name = f"PPMI_{subject_id}_{folder_name}_S{study_id}_I{image_id}.xml"
            xml_file_path = os.path.join(meta_data_path, xml_file_name)
            
            # Check if the XML file exists
            if os.path.exists(xml_file_path):
                # Extract the label from the XML
                label = extract_label_from_xml(xml_file_path)
                if label is not None:
                    # Add data to the CSV rows
                    csv_data.append([label, subject_id, nii_file_path])
            else:
                print(f"Warning: XML file not found for {nii_file_path}")
                print(f"Expected XML path: {xml_file_path}")

# Write the collected data to the CSV file
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["label", "subject_id", "nii_file_path"])  # Write header
    writer.writerows(csv_data)  # Write data rows

# Print the label mapping for reference
print("Label mapping:", label_mapping)
print(f"CSV file created: {output_csv}")
