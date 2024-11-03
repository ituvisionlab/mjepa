import csv

def change_absolute_paths(input_csv, output_csv):
    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write the header row
        header = next(reader)
        writer.writerow(header)
        
        # Filter out rows with "Spatially_Normalized" in the path
        for row in reader:
            # Check if "Spatially_Normalized" is not present in the file path
            row[2] = row[2].replace("/media/yusuf/backup", "/media/disk2")
            writer.writerow(row)
    
    print(f"CSV file with changed paths created: {output_csv}")

# Example usage
input_csv = "filtered_combined_nii_file_paths_256slices.csv"  # Path to the combined CSV file
output_csv = "filtered_combined_nii_file_paths_256slices_new.csv"  # Path to the filtered CSV file
change_absolute_paths(input_csv, output_csv)
