import os
import csv

def generate_csv_for_nii_volumes(nii_folder, csv_file_path):
    """
    Generates a CSV file with columns: label, subject_id, nii_file_path.

    Args:
        nii_folder (str): Path to the folder containing NIfTI files.
        csv_file_path (str): Path to the output CSV file.
    """
    # List all files in the nii_folder
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith('.nii')]

    # Prepare data for CSV
    csv_data = []
    for nii_file in nii_files:
        # Extract idx and label from the filename
        # Filename format: volume_{idx}_label_{label}.nii
        base_name = os.path.splitext(nii_file)[0]  # Remove .nii extension
        parts = base_name.split('_')
        try:
            idx = parts[1]
            label = parts[3]
        except IndexError:
            print(f"Filename {nii_file} does not match expected format.")
            continue

        # Generate subject_id (you can customize this as needed)
        subject_id = f"MNIST_{idx.zfill(5)}"

        # Construct the full path to the NIfTI file
        nii_file_path = os.path.abspath(os.path.join(nii_folder, nii_file))

        # Append the data to the list
        csv_data.append([label, subject_id, nii_file_path])

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(['label', 'subject_id', 'nii_file_path'])
        # Write the data rows
        for row in csv_data:
            csv_writer.writerow(row)

    print(f"CSV file saved at {csv_file_path}")

def main():
    # Existing code...
    # After saving the NIfTI files, generate the CSV file

    nii_folder = '/gpfs/data/sodicksonlab/gozde/mnist3d/nii_volumes'  # Folder containing the NIfTI files
    csv_file_path = '/gpfs/home/unalg01/jepa/src/datasets/mnist3d/nii_volumes.csv'  # Path to the output CSV file

    generate_csv_for_nii_volumes(nii_folder, csv_file_path)

if __name__ == '__main__':
    main()
