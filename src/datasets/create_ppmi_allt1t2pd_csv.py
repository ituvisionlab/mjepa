import os
import csv
import glob
import nibabel as nib
import xml.etree.ElementTree as ET

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
# Function to filter NIfTI files based on FOV and spacing
def filter_nifti(img, min_fov=50, max_spacing=6.5):
    header = img.header
    voxel_spacing = header.get_zooms()[:3]
    image_dimensions = header.get_data_shape()[:3]
    fov = [spacing * dim for spacing, dim in zip(voxel_spacing, image_dimensions)]
    if any(f < min_fov for f in fov) or any(s > max_spacing for s in voxel_spacing):
        return False
    return True

def extract_info_from_xml(xml_path):
    try:
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Define the namespace (if applicable)
        ns = {'ns': 'http://ida.loni.usc.edu'}

        # Extract <researchGroup> under <subject> and <project>
        label = None
        research_group = root.find(".//ns:project/ns:subject/ns:researchGroup", ns)
        if research_group is None:  # Fallback without namespace
            research_group = root.find(".//project/subject/researchGroup")

        if research_group is not None:
            label_text = research_group.text.strip()
            if label_text not in label_mapping:
                label_mapping[label_text] = len(label_mapping)
            label = label_mapping[label_text]

        # Extract <protocol term="Weighting"> under <protocolTerm>
        contrast = None
        for protocol in root.findall(".//ns:protocolTerm/ns:protocol", ns):
            if protocol.attrib.get("term") == "Weighting":
                contrast = protocol.text.strip()
                break

        if contrast is None:  # Fallback without namespace
            for protocol in root.findall(".//protocolTerm/protocol"):
                if protocol.attrib.get("term") == "Weighting":
                    contrast = protocol.text.strip()
                    break

        # Extract <dateAcquired>, <subjectSex>, and <subjectAge>
        date_acquired = root.find(".//ns:series/ns:dateAcquired", ns)
        if date_acquired is None:  # Fallback without namespace
            date_acquired = root.find(".//series/dateAcquired")
        date_acquired = date_acquired.text.strip() if date_acquired is not None else "Unknown"

        subject_sex = root.find(".//ns:subject/ns:subjectSex", ns)
        if subject_sex is None:  # Fallback without namespace
            subject_sex = root.find(".//subject/subjectSex")
        subject_sex = subject_sex.text.strip() if subject_sex is not None else "Unknown"

        subject_age = root.find(".//ns:study/ns:subjectAge", ns)
        if subject_age is None:  # Fallback without namespace
            subject_age = root.find(".//study/subjectAge")
        subject_age = subject_age.text.strip() if subject_age is not None else "Unknown"

        return label, contrast, date_acquired, subject_sex, subject_age

    except Exception as e:
        print(f"Error reading XML {xml_path}: {e}")
        return None, None, "Unknown", "Unknown", "Unknown"

# Collect all rows for the CSV
csv_data = []
cntr = 0
cntr_size_filtered = 0
cntr_filtered = 0

# Walk through the dataset folders
for root, dirs, files in os.walk(base_data_path):
    for file in files:
        if file.endswith(".nii") and "mask" not in file.lower():
            cntr += 1
            print(cntr)

            nii_file_path = os.path.join(root, file)

            # Skip small files
            if os.path.getsize(nii_file_path) < 2.5e6:
                print(f"Skipped small file: {nii_file_path}")
                cntr_size_filtered += 1
                continue

            # Load the NIfTI image and apply FOV and spacing filter
            try:
                img = nib.load(nii_file_path)
                if not filter_nifti(img):
                    print(f"Excluded based on FOV/spacing: {nii_file_path}")
                    cntr_filtered += 1
                    continue
            except Exception as e:
                print(f"Error loading NIfTI file {nii_file_path}: {e}")
                continue

            # Extract subject ID, modality, study date, and image ID from the file path
            path_parts = nii_file_path.split(os.sep)
            subject_id = path_parts[-5]  # e.g., "60036"
            modality = path_parts[-4]  # e.g., "T2_in_T1-anatomical_space"
            study_date = path_parts[-3]  # e.g., "2014-03-04_11_26_56.0"
            image_id = path_parts[-2].lstrip("I")  # e.g., "451800"

            # Extract study ID from the .nii file name
            study_id = file.split('_S')[-1].split('_I')[0]  # Extract study ID (e.g., "225653")

            # Construct the XML file path using glob for robustness
            xml_pattern = f"PPMI_{subject_id}_{modality}_S{study_id}_I{image_id}.xml"
            xml_file_path = glob.glob(os.path.join(meta_data_path, xml_pattern))

            if xml_file_path:
                xml_file_path = xml_file_path[0]  # Take the first match
                label, contrast, date_acquired, subject_sex, subject_age = extract_info_from_xml(xml_file_path)
                if label is not None and contrast:  # Include all contrasts
                    csv_data.append([label, subject_id, contrast, date_acquired, subject_sex, subject_age, nii_file_path])
            else:
                print(f"Warning: XML file not found for {nii_file_path}")
                print(f"Expected XML pattern: {xml_pattern}")

# Update contrast labels for NIfTI files containing "T2_in_T1" in their paths
for row in csv_data:
    nii_file_path = row[-1]  # Get the NIfTI file path (last column in the row)
    if "T2_in_T1" in nii_file_path:
        row[2] = "T2"  # Update the contrast label (third column in the row)

# Write to CSV
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["label", "subject_id", "contrast", "date_acquired", "subject_sex", "subject_age", "nii_file_path"])  # Write header
    writer.writerows(csv_data)  # Write data rows

# Print the label mapping for reference
print("Label mapping:", label_mapping)
print(f"CSV file created: {output_csv}")
print(f"Total NIfTI files processed: {len(csv_data)}")
print(f"Total NIfTI files size filtered: {cntr_size_filtered}")
print(f"Total NIfTI files fov/spacing filtered: {cntr_filtered}")
