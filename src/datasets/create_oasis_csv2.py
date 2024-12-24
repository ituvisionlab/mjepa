import os
import glob
import csv
import pandas as pd

# Read the cognitive data from the CSV file
oasis_df = pd.read_csv("/gpfs/home/unalg01/jepa/src/datasets/OASIS3_UDSb4_cdr.csv")

# Filter columns to include only those we need
oasis_df = oasis_df[["OASISID", "days_to_visit", "CDRTOT"]]

# Get all NIfTI files recursively
files = glob.glob("/gpfs/data/sodicksonlab/gozde/OASIS3/**/*.nii.gz", recursive=True)
files = sorted(files)

# Open a new CSV file for writing
with open("/gpfs/home/unalg01/jepa/src/datasets/oasis_nii.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the header for the CSV file
    writer.writerow(["label", "subject_id", "nii_file_path"])
    
    for nii_file in files:
        # Extract the subject ID from the folder name
        folder_name = nii_file.split("/")[-3]  # e.g., "OAS30668_MR_d1153"
        subject_id = folder_name.split("_")[0]  # e.g., "OAS30668"
        
        # Extract the scan day from the folder name
        scan_day_str = folder_name.split("_")[-1]  # e.g., "d1153"
        if scan_day_str.startswith("d"):
            scan_day = int(scan_day_str[1:])  # Extract the number after 'd'
        else:
            print(f"Warning: Unable to extract scan day from folder name: {folder_name}")
            continue

        # Filter the cognitive data for the current subject
        subject_df = oasis_df[oasis_df["OASISID"] == subject_id]
        if subject_df.empty:
            print(f"Warning: No cognitive data found for subject {subject_id}. Skipping.")
            continue

        # Find the closest cognitive exam day
        subject_df["day_diff"] = abs(subject_df["days_to_visit"] - scan_day)
        closest_row = subject_df.loc[subject_df["day_diff"].idxmin()]

        # Get the CDRTOT score as the label
        label = closest_row["CDRTOT"]

        # Write the entry to the CSV
        writer.writerow([label, subject_id, nii_file])

print("CSV file created successfully.")
