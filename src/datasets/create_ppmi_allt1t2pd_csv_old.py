import os
import csv

# Paths to PPMI dataset and metadata
base_data_path = "/gpfs/data/sodicksonlab/gozde/PPMI"
meta_data_path = "/gpfs/data/sodicksonlab/gozde/PPMI-meta/PPMI"

# Output CSV file
output_csv = "ppmi_all_nii.csv"

# Label mapping: {'Control': 0, 'PD': 1, 'Prodromal': 2, 'SWEDD': 3, 'Other': 4}
label_mapping = {
    "Control": 0,
    "PD": 1,
    "Prodromal": 2,
    "SWEDD": 3,
    "Other": 4
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

# Collect all rows for the CSV
csv_data = []

# Walk through the dataset folders
for root, dirs, files in os.walk(base_data_path):
    for file in files:
        if file.endswith(".nii"):
            # Construct the absolute path to the .nii file
            nii_file_path = os.path.join(root, file)
            
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
