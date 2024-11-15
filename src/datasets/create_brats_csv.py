import os
import csv

train_data_path = "src/datasets/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
val_data_path = "src/datasets/BraTS2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"

train_folder_list = os.listdir(train_data_path)
train_folder_list = [os.path.join(train_data_path, i) for i in train_folder_list]

val_folder_list = os.listdir(val_data_path)
val_folder_list = [os.path.join(val_data_path, i) for i in val_folder_list]

train_folder_list += val_folder_list

with open("src/datasets/brats_file_paths.csv", mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
        
    # Write the header for the CSV file
    writer.writerow(['label', 'subject_id', 'nii_file_path'])
    
    for folder in train_folder_list:
        if ".csv" in folder:
            continue
        nii_files = os.listdir(folder)
        
        t1_nii_file = list(filter(lambda x: "t1.nii" in x, nii_files))[0]
        
        subject_id = folder.split("_")[-1]        
        t1_nii_file_path = os.path.join(folder, t1_nii_file)
        
        writer.writerow([0, subject_id, t1_nii_file_path])

print("CSV file created")

        
        
        