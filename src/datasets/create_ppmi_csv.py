import os
import csv

# Paths to PPMI dataset and metadata
base_data_path = "/gpfs/data/sodicksonlab/gozde/PPMI"
meta_data_path = "/gpfs/data/sodicksonlab/gozde/PPMI-meta/PPMI"

# Output CSV file
output_csv = "ppmi_T1_anatomical_nii.csv"

# Label mapping: {'Control': 0, 'PD': 1, 'Prodromal': 2, 'SWEDD': 3, 'Other': 4}
# Define a label mapping
label_mapping = {
    "Control": 0,  # Healthy Controls
    "PD": 1,  # Parkinson's Disease
    "Prodromal": 2,  # Prodromal Parkinson's Disease
    "SWEDD": 3,  # Scans without evidence of dopaminergic deficit
    "Other": 4  # For any other unknown label
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
    # Skip any folders containing "T2_in_T1-anatomical_space"
    if "T2_in_T1-anatomical_space" in root:
        continue
    
    # Only process paths containing "T1-anatomical"
    if "T1-anatomical" in root:
        for file in files:
            if file.endswith(".nii"):
                # Construct the absolute path to the .nii file
                nii_file_path = os.path.join(root, file)
                
                # Extract subject ID and study ID from the .nii file name
                nii_filename = os.path.basename(file)
                parts = nii_filename.split("_")
                subject_id = parts[1]  # Extract subject ID
                study_id = [part for part in parts if part.startswith("S")][0]  # Extract study ID
                image_id = parts[-1].split(".")[0][1:]  # Extract image ID (e.g., I409169 -> 409169)
                
                # Construct the corresponding XML file path
                xml_file_name = f"PPMI_{subject_id}_T1-anatomical_{study_id}_I{image_id}.xml"
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
