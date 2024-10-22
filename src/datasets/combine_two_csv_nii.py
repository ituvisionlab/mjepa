import csv

def combine_csv_files(nc_csv, ad_csv, output_csv):
    # Open the output CSV file for writing
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header for the CSV file
        writer.writerow(['label', 'subject_id', 'nii_file_path'])
        
        # Function to filter out spatially normalized files
        def is_spatially_normalized(row):
            # Ensure the row has at least 3 columns before checking
            return len(row) > 2 and "Spatially_Normalized" in row[2]
        
        # Read and write data from the NC CSV file with label 0
        with open(nc_csv, mode='r') as nc_file:
            reader = csv.reader(nc_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_MINIMAL)
            next(reader)  # Skip header
            for row in reader:
                if len(row) > 2 and not is_spatially_normalized(row):
                    writer.writerow([0] + row)  # Add label 0 and write the row
                
        # Read and write data from the AD CSV file with label 1
        with open(ad_csv, mode='r') as ad_file:
            reader = csv.reader(ad_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_MINIMAL)
            next(reader)  # Skip header
            for row in reader:
                if len(row) > 2 and not is_spatially_normalized(row):
                    writer.writerow([1] + row)  # Add label 1 and write the row
    
    print(f"Filtered and combined CSV file created: {output_csv}")

# Example usage
nc_csv = "nii_file_paths_NC.csv"  # Path to the NC CSV file
ad_csv = "nii_file_paths_AD.csv"  # Path to the AD CSV file
output_csv = "filtered_combined_nii_file_paths.csv"  # Path to the output CSV file
combine_csv_files(nc_csv, ad_csv, output_csv)
