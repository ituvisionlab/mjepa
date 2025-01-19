import csv

# Input and output file paths
input_csv = "oasis3_all_nii_original.csv"  # Replace with your actual input CSV file name
output_csv = "oasis3_all_nii.csv"  # Output CSV file with new fields

# Placeholder values
default_date = "2000-01-01"
default_sex = "X"
default_age = "0.00"

# Read the input CSV and write the updated output CSV
with open(input_csv, mode="r", newline="") as infile, open(output_csv, mode="w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Read and modify the header
    header = next(reader)
    new_header = header[:3] + ["date_acquired", "subject_sex", "subject_age"] + [header[3]]
    writer.writerow(new_header)
    
    # Process and write each row with placeholder values
    for row in reader:
        updated_row = row[:3] + [default_date, default_sex, default_age] + [row[3]]
        writer.writerow(updated_row)

print(f"Updated CSV file created: {output_csv}")
