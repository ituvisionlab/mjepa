import os
import csv

train_data_path = "/gpfs/data/sodicksonlab/gozde/MOOD/brain_train"

train_folder_list = sorted(os.listdir(train_data_path))
train_folder_list = [os.path.join(train_data_path, i) for i in train_folder_list]

with open("src/datasets/mood_file_paths.csv", mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
        
    # Write the header for the CSV file
    writer.writerow(['label', 'subject_id', 'nii_file_path'])
    
    for nii_file in train_folder_list:
        
        subject_id = nii_file.split("/")[-1].split(".")[0]        
        t1_nii_file_path = nii_file
        
        writer.writerow([0, subject_id, t1_nii_file_path])

print("CSV file created")

        
        
        