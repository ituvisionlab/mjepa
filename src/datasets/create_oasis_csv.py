import csv
import glob
import os
import pandas as pd

# Load the OASIS CSV file containing cognitive exam information
oasis_csv_path = "/gpfs/home/unalg01/jepa/src/datasets/OASIS3_UDSb4_cdr.csv"
oasis_df = pd.read_csv(oasis_csv_path)

# Extract columns we care about
oasis_df = oasis_df[["OASISID", "days_to_visit", "CDRTOT"]]

# Ensure the "days_to_visit" column is numeric for comparison
oasis_df["days_to_visit"] = pd.to_numeric(oasis_df["days_to_visit"], errors="coerce")

# Collect all .nii.gz files
files = glob.glob("/gpfs/data/sodicksonlab/gozde/OASIS3/**/*.nii.gz", recursive=True)
files = sorted(files)

# Prepare the output CSV
output_csv_path = "/gpfs/home/unalg01/jepa/src/datasets/oasis_nii.csv"
with open(output_csv_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    # Write the header for the CSV file
    writer.writerow(["label", "subject_id", "nii_file_path"])

    for nii_file in files:
        # Extract the subject ID and scan day from the file path
        subject_id = nii_file.split("/")[-1]
        scan_day_str = nii_file.split("_")[-1]  # Example: "d0129"
        scan_day = int(scan_day_str[1:])  # Extract number after 'd'

        # Filter the cognitive data for the current subject
        subject_df = oasis_df[oasis_df["OASISID"] == subject_id]

        if subject_df.empty:
            print(f"Warning: No cognitive data found for subject {subject_id}. Skipping.")
            continue

        # Find the closest cognitive exam day
        subject_df["day_diff"] = abs(subject_df["days_to_visit"] - scan_day)
        closest_row = subject_df.loc[subject_df["day_diff"].idxmin()]

        # Get the CDRTOT score
        label = closest_row["CDRTOT"]

        # Write the entry to the CSV
        writer.writerow([label, subject_id, nii_file])

print("CSV file with labels created successfully!")
