import os
import csv
import glob

files = glob.glob("/media/disk2/OASIS3/**/*.nii.gz", recursive=True)

files = sorted(files)

with open("src/datasets/oasis_file_paths.csv", mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
        
    # Write the header for the CSV file
    writer.writerow(['label', 'subject_id', 'nii_file_path'])
    
    for nii_file in files:
        
        subject_id = nii_file.split("/")[4]     
        t1_nii_file_path = nii_file
        
        writer.writerow([0, subject_id, t1_nii_file_path])

print("CSV file created")

        
        
        