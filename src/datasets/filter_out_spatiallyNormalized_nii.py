import csv

def filter_spatially_normalized(input_csv, output_csv):
    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write the header row
        header = next(reader)
        writer.writerow(header)
        
        # Filter out rows with "Spatially_Normalized" in the path
        for row in reader:
            # Check if "Spatially_Normalized" is not present in the file path
            if "Spatially_Normalized" not in row[2]:
                writer.writerow(row)
    
    print(f"Filtered CSV file created: {output_csv}")

# Example usage
input_csv = "combined_nii_file_pathsAll.csv"  # Path to the combined CSV file
output_csv = "filtered_combined_nii_file_paths.csv"  # Path to the filtered CSV file
filter_spatially_normalized(input_csv, output_csv)
