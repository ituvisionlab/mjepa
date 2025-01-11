import os
import csv
import nibabel as nib

# Paths to ADNI dataset and metadata
base_data_path = "/gpfs/data/sodicksonlab/gozde/ADNIall/ADNI"
meta_data_path = "/gpfs/data/sodicksonlab/gozde/ADNI-meta/ADNI"

# Output CSV file
output_csv = "adni_allt1_nii.csv"

# Label mapping
label_mapping = {
    "CN": 0,
    "MCI": 1,
    "AD": 2,
    "Other": 3
}

# Function to extract label and contrast from XML
def extract_info_from_xml(xml_path):
    try:
        with open(xml_path, "r") as file:
            content = file.read()
            # Extract <researchGroup> (label)
            start_tag = "<researchGroup>"
            end_tag = "</researchGroup>"
            start_idx = content.find(start_tag)
            end_idx = content.find(end_tag)
            label = None
            if start_idx != -1 and end_idx != -1:
                label_text = content[start_idx + len(start_tag):end_idx].strip()
                if label_text not in label_mapping:
                    label_mapping[label_text] = len(label_mapping)
                label = label_mapping[label_text]
            
            # Extract <protocol term="Weighting"> (contrast)
            protocol_tag = '<protocol term="Weighting">'
            protocol_end_tag = "</protocol>"
            protocol_start = content.find(protocol_tag)
            protocol_end = content.find(protocol_end_tag)
            contrast = None
            if protocol_start != -1 and protocol_end != -1:
                contrast = content[protocol_start + len(protocol_tag):protocol_end].strip()
            
            return label, contrast
    except Exception as e:
        print(f"Error reading XML {xml_path}: {e}")
        return None, None

# Function to filter NIfTI files based on FOV and spacing
def filter_nifti(img, min_fov=50, max_spacing=6.5):
    header = img.header
    voxel_spacing = header.get_zooms()[:3]
    image_dimensions = header.get_data_shape()[:3]
    fov = [spacing * dim for spacing, dim in zip(voxel_spacing, image_dimensions)]
    if any(f < min_fov for f in fov) or any(s > max_spacing for s in voxel_spacing):
        return False
    return True

# Collect all rows for the CSV
csv_data = []
cntr = 0

# Walk through the dataset folders
for root, dirs, files in os.walk(base_data_path):
    for file in files:
        if file.endswith(".nii"):
            cntr += 1
            print(cntr)
            # Skip files with "Mask" in their names
            if "Mask" in file:
                continue
            
            nii_file_path = os.path.join(root, file)
            
            # Skip small files
            if os.path.getsize(nii_file_path) < 2e6:
                continue
            
            img = nib.load(nii_file_path)
            if not filter_nifti(img):
                continue
            
            # Extract subject ID, study ID, and image ID
            path_parts = nii_file_path.split(os.sep)
            subject_id = path_parts[-5]
            folder_name = path_parts[-4]
            study_date = path_parts[-3]
            image_id = path_parts[-2].lstrip("I")
            study_id = file.split('_S')[-1].split('_I')[0]
            
            # Construct the XML file path
            xml_file_name = f"ADNI_{subject_id}_{folder_name}_S{study_id}_I{image_id}.xml"
            xml_file_path = os.path.join(meta_data_path, xml_file_name)
            
            # Extract label and contrast
            if os.path.exists(xml_file_path):
                label, contrast = extract_info_from_xml(xml_file_path)
                if label is not None and contrast == "T1":  # Include only T1-weighted files
                    csv_data.append([label, subject_id, contrast, nii_file_path])
            else:
                print(f"Warning: XML file not found for {nii_file_path}")
                print(f"Expected XML path: {xml_file_path}")

# Write to CSV
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["label", "subject_id", "contrast", "nii_file_path"])
    writer.writerows(csv_data)

# Print the label mapping for reference
print("Label mapping:", label_mapping)
print(f"CSV file created with all contrasts excluding masks: {output_csv}")
