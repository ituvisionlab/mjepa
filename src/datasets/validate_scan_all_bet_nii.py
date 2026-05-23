import pandas as pd
import os
import nibabel as nib
import numpy as np

# Input and output CSV paths
CSV_PATH = "/gpfs/home/unalg01/jepa/src/datasets/SCAN_NIFTI_all_with_cleaned_contrast.csv"
OUTPUT_CSV = "/gpfs/home/unalg01/jepa/src/datasets/SCAN_NIFTI_all_final.csv"

# CSV_PATH = "SCAN_NIFTI_all_with_bbox.csv"
# OUTPUT_CSV = "SCAN_NIFTI_all_with_betmask_and_bbox.csv"

# Read the original CSV
df = pd.read_csv(CSV_PATH)
updated_rows = []

for i, row in df.iterrows():
    orig_path = row["nii_file_path"]
    if not os.path.exists(orig_path):
        print(f"[WARNING] Original file missing: {orig_path}")
        continue

    # Construct paths
    betmask_path = orig_path
    betmask_mask_path = orig_path.replace(".nii.gz", "_mask.nii.gz")
    # betmask_path = orig_path.replace(".nii.gz", "_betmask.nii.gz") #the above csv already has betmask.nii in the file names
    # betmask_mask_path = orig_path.replace(".nii.gz", "_betmask_mask.nii.gz")

    # Verify both files exist
    if not os.path.exists(betmask_path):
        print(f"[WARNING] Missing _betmask file: {betmask_path}")
        continue
    if not os.path.exists(betmask_mask_path):
        print(f"[WARNING] Missing _betmask_mask file: {betmask_mask_path}")
        continue

    try:
        # Load the binary brain mask
        mask_img = nib.load(betmask_mask_path)
        mask_data = mask_img.get_fdata()

        # Check if the mask contains any brain voxels
        if not np.any(mask_data):
            print(f"[INFO] Empty mask (no brain voxels): {betmask_mask_path}")
            continue

        # Compute bounding box from nonzero mask indices
        coords = np.array(np.nonzero(mask_data)) # coords.shape = (3, N), N coordinates in 3 axes
        xmin, ymin, zmin = coords.min(axis=1)
        xmax, ymax, zmax = coords.max(axis=1)
        # zmin, ymin, xmin = coords.min(axis=1)
        # zmax, ymax, xmax = coords.max(axis=1)

        # Update fields
        row["nii_file_path"] = betmask_path  # use skull-stripped image in CSV
        row["xmin"] = int(xmin)
        row["xmax"] = int(xmax)
        row["ymin"] = int(ymin)
        row["ymax"] = int(ymax)
        row["zmin"] = int(zmin)
        row["zmax"] = int(zmax)

        updated_rows.append(row)

    except Exception as e:
        print(f"[ERROR] Failed on {betmask_mask_path}: {e}")
        continue

# Create updated DataFrame and write to CSV
df_updated = pd.DataFrame(updated_rows)
df_updated.to_csv(OUTPUT_CSV, index=False)
print(f"[DONE] Updated CSV with betmask paths and bbox saved to:\n{OUTPUT_CSV}")
