import os
import csv
import nibabel as nib
import glob
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

def extract_info_from_xml(xml_path):
    try:
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Define the namespace (if needed, depending on your XML structure)
        ns = {'ns': 'http://ida.loni.usc.edu'}

        # Extract <subjectIdentifier> (subject ID)
        subject_id_elem = root.find(".//ns:subject/ns:subjectIdentifier", ns)
        if subject_id_elem is None:  # Try without namespace if the above fails
            subject_id_elem = root.find(".//subject/subjectIdentifier")
        subject_id = subject_id_elem.text.strip() if subject_id_elem is not None else "Unknown"

        # Extract <researchGroup> (label)
        label = None
        research_group = root.find(".//ns:project/ns:subject/ns:researchGroup", ns)
        if research_group is None:  # Try without namespace if the above fails
            research_group = root.find(".//project/subject/researchGroup")
        if research_group is not None:
            label_text = research_group.text.strip()
            if label_text not in label_mapping:
                label_mapping[label_text] = len(label_mapping)
            label = label_mapping[label_text]

        # Extract "Weighting" term from <protocol>
        contrast = None
        for protocol in root.findall(".//protocol"):
            if protocol.attrib.get("term") == "Weighting":
                contrast = protocol.text.strip()
                break

        # Extract <dateAcquired>, <subjectSex>, <subjectAge>, and <weightKg>
        date_acquired = root.find(".//ns:series/ns:dateAcquired", ns)
        if date_acquired is None:
            date_acquired = root.find(".//series/dateAcquired")
        date_acquired = date_acquired.text.strip() if date_acquired is not None else "Unknown"

        subject_sex = root.find(".//ns:subject/ns:subjectSex", ns)
        if subject_sex is None:
            subject_sex = root.find(".//subject/subjectSex")
        subject_sex = subject_sex.text.strip() if subject_sex is not None else "Unknown"

        subject_age = root.find(".//ns:study/ns:subjectAge", ns)
        if subject_age is None:
            subject_age = root.find(".//study/subjectAge")
        subject_age = subject_age.text.strip() if subject_age is not None else "Unknown"
        
        subject_weight = root.find(".//ns:study/ns:weightKg", ns)
        if subject_weight is None:
            subject_weight = root.find(".//study/weightKg")
        subject_weight = subject_weight.text.strip() if subject_weight is not None else "Unknown"

        return label, subject_id, contrast, date_acquired, subject_sex, subject_age, subject_weight

    except Exception as e:
        print(f"Error reading XML {xml_path}: {e}")
        return None, "Unknown", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"


# Function to filter NIfTI files based on FOV and spacing
def filter_nifti(img, file_path, min_fov=50, max_spacing=6.5):
    header = img.header
    volume = img.get_fdata()
    # Return false if the volume is not 3D
    if volume.ndim != 3:
        print(f'Skipping non-3D volume (dim={volume.ndim}) at: {file_path}')
        return False
    voxel_spacing = header.get_zooms()[:3]
    image_dimensions = header.get_data_shape()[:3]
    fov = [spacing * dim for spacing, dim in zip(voxel_spacing, image_dimensions)]
    if any(f < min_fov for f in fov) or any(s > max_spacing for s in voxel_spacing):
        return False
    return True


# Collect all rows for the CSV
csv_data = []
cntr = 0
cntr_size_filtered = 0
cntr_filtered = 0
unique_subject_ids = set()  # Initialize a set to store unique subject IDs

# Walk through the dataset folders
for root, dirs, files in os.walk(base_data_path):
    for file in files:
        if file.endswith(".nii") and "mask" not in file.lower():
            cntr += 1
            print(f"Processing file {cntr}: {file}")
            
            nii_file_path = os.path.join(root, file)
            
            # Skip small files
            if os.path.getsize(nii_file_path) < 2.5e6:
                print(f"Skipped small file: {nii_file_path}")
                cntr_size_filtered += 1
                continue
            
            # Load the NIfTI image and apply dimension, FOV and spacing filter
            try:
                img = nib.load(nii_file_path)
                if not filter_nifti(img, nii_file_path):
                    print(f"Excluded based on FOV/spacing and non3D volume: {nii_file_path}")
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
                label, subject_id, contrast, date_acquired, subject_sex, subject_age, subject_weight = extract_info_from_xml(xml_file_path)
                if label is not None and contrast:  # Include all contrasts
                    csv_data.append([label, subject_id, contrast, date_acquired, subject_sex, subject_age, subject_weight, nii_file_path])
                    unique_subject_ids.add(subject_id)  # Add subject_id to the set
            else:
                print(f"Warning: XML file not found for {nii_file_path}")
                print(f"Expected XML pattern: {xml_pattern}")

# Write to CSV
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["label", "subject_id", "contrast", "date_acquired", "subject_sex", "subject_age", "subject_weight", "nii_file_path"])
    writer.writerows(csv_data)


# Log file to write summary information
log_file_path = "ppmi_dataset_summary.log"

# Write summary information to the log file
with open(log_file_path, mode="w") as log_file:
    log_file.write(f"Total number of distinct subjects in the dataset: {len(unique_subject_ids)}\n")
    log_file.write(f"Label mapping: {label_mapping}\n")
    log_file.write(f"CSV file created with all contrasts excluding masks: {output_csv}\n")
    log_file.write(f"Total NIfTI files processed: {len(csv_data)}\n")
    log_file.write(f"Total NIfTI files size filtered: {cntr_size_filtered}\n")
    log_file.write(f"Total NIfTI files FOV/spacing filtered: {cntr_filtered}\n")

print(f"Summary information written to {log_file_path}")
