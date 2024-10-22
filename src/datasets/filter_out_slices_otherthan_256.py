import pandas as pd
import nibabel as nib
import os

def filter_nifti_files(csv_file, output_csv_file, desired_slice_size=256, slice_axis=2):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_file)
    
    # Initialize a list to hold indices of rows to keep
    rows_to_keep = []
    
    # Iterate over each row
    for index, row in data.iterrows():
        label = row['label']
        subject_id = row['subject_id']
        nii_file_path = row['nii_file_path']
        
        # Check if the file exists
        if not os.path.exists(nii_file_path):
            print(f"File not found: {nii_file_path}")
            continue  # Skip this file
        
        try:
            # Load the NIfTI file
            img = nib.load(nii_file_path)
            data_shape = img.get_fdata().shape
            print(f"Subject ID: {subject_id}, Shape: {data_shape}")
            
            # Check the slice size along the specified axis
            slice_size = data_shape[slice_axis]
            
            print(f"Slice size: {slice_size}")

            if slice_size == desired_slice_size:
                rows_to_keep.append(index)
            else:
                print(f"Excluding subject {subject_id} with slice size {slice_size} along axis {slice_axis}")
        except Exception as e:
            print(f"Error loading file {nii_file_path}: {e}")
            continue  # Skip this file
    
    # Create a new DataFrame with only the rows to keep
    filtered_data = data.loc[rows_to_keep]
    
    # Save the filtered DataFrame to a new CSV file
    filtered_data.to_csv(output_csv_file, index=False)
    print(f"Filtered data saved to {output_csv_file}")

if __name__ == "__main__":
    csv_file = 'filtered_combined_nii_file_paths.csv'
    output_csv_file = 'filtered_combined_nii_file_paths_256slices.csv'
    desired_slice_size = 256
    slice_axis = 2  # Adjust if necessary after checking axis correspondence

    filter_nifti_files(csv_file, output_csv_file, desired_slice_size, slice_axis)
