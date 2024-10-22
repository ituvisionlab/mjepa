import csv
import pandas as pd

# Input and output CSV file paths
input_csv = '/home/gozde/medChangeDet/jepa/src/datasets/filtered_combined_nii_file_paths_256slices.csv'
output_csv = '/home/gozde/medChangeDet/jepa/src/datasets/formatted_nii_file_paths.csv'

# Open the input and output files
with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
    
    # Write the header line in the output CSV
    writer.writerow(['nii_file_path', 'label'])
    
    # Skip the header in the input file
    next(reader)
    
    # Iterate through each row in the input CSV
    for row in reader:
        label = row[0]  # Integer label
        nii_file_path = row[2]  # Absolute file path
        
        # Write to output CSV in the required format (file_absolute_path SPACE integer label)
        writer.writerow([nii_file_path, label])

print(f"Formatted CSV file created with header: {output_csv}")

# test reading from the new file
samples, labels = [], []
      
data = pd.read_csv(output_csv)
samples += data['nii_file_path'].tolist()
labels += data['label'].tolist()

print(samples)
print(labels)