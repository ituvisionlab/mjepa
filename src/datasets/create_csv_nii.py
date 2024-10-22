import os
import csv
import hashlib

def compute_md5(file_path):
    """Compute the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def write_nii_paths_to_csv(parent_dirs, output_csv):
    # Set to keep track of added files by their MD5 hashes
    seen_files = set()

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header for the CSV file
        writer.writerow(['subject_id', 'nii_file_path'])
        
        # Iterate over each parent directory in the list
        for parent_dir in parent_dirs:
            # Iterate through each subject folder directly under the parent directory
            for subject_id in os.listdir(parent_dir):
                subject_path = os.path.join(parent_dir, subject_id)
                
                # Ensure we're only processing directories (i.e., subject folders)
                if os.path.isdir(subject_path):
                    # Walk through the subject folder and find all .nii files
                    for root, dirs, files in os.walk(subject_path):
                        for file in files:
                            if file.endswith(".nii"):  # Check if it's a .nii file
                                # Create the full file path
                                file_path = os.path.join(root, file)
                                
                                # Compute the MD5 checksum to detect duplicates
                                file_md5 = compute_md5(file_path)
                                
                                # If the file's MD5 checksum is already seen, skip it
                                if file_md5 in seen_files:
                                    continue
                                
                                # Add the file's MD5 checksum to the set
                                seen_files.add(file_md5)
                                
                                # Write the subject_id and file path to the CSV
                                writer.writerow([subject_id, file_path])
    
    print(f"CSV file created: {output_csv}")

# Example usage
parent_dirs = [
    "/media/yusuf/backup/ADNI-NC/ADNI_NC",
    "/media/yusuf/backup/ADNI-NC/ADNI_NC1",
    "/media/yusuf/backup/ADNI-NC/ADNI_NC2",
    "/media/yusuf/backup/ADNI-NC/ADNI_NC3",
    "/media/yusuf/backup/ADNI-NC/ADNI_NC4"
]  # List of all directories where the NIfTI files are located
#parent_dirs = [
#    "/media/yusuf/backup/ADNI2-AD/ADNI_AD",
#    "/media/yusuf/backup/ADNI2-AD/ADNI_AD1",
#    "/media/yusuf/backup/ADNI2-AD/ADNI_AD2",
#    "/media/yusuf/backup/ADNI2-AD/ADNI_AD3",
#    "/media/yusuf/backup/ADNI2-AD/ADNI_AD4"
#]  # List of all directories where the NIfTI files are located
output_csv = "nii_file_paths.csv"  # Path to the output CSV file
write_nii_paths_to_csv(parent_dirs, output_csv)
