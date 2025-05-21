import os
import pandas as pd
import nibabel as nib
import numpy as np

# Paths
input_csv = "/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_nii_with_bbox.csv"
cleaned_csv = "/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_nii_verified.csv"

# Load the CSV
df = pd.read_csv(input_csv)

# Columns to verify/update
bbox_columns = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
missing_files = []
updated_rows = 0

# Bounding box computation
def calculate_bbox(mask_path):
    try:
        nii = nib.load(mask_path)
        data = nii.get_fdata()
        coords = np.argwhere(data > 0)
        if coords.size == 0:
            return None
        return {
            "xmin": coords[:, 0].min(),
            "xmax": coords[:, 0].max(),
            "ymin": coords[:, 1].min(),
            "ymax": coords[:, 1].max(),
            "zmin": coords[:, 2].min(),
            "zmax": coords[:, 2].max(),
        }
    except Exception as e:
        print(f"Error loading {mask_path}: {e}")
        return None

# Iterate over rows
for idx, row in df.iterrows():
    nii_path = row["nii_file_path"]

    if not os.path.exists(nii_path):
        print(f"Missing file: {nii_path}")
        missing_files.append(nii_path)
        continue

    # Recompute bounding box
    bbox = calculate_bbox(nii_path)
    if bbox:
        for col in bbox_columns:
            df.at[idx, col] = bbox[col]
        updated_rows += 1
    else:
        try:
            img = nib.load(nii_path)
            x, y, z = img.shape[:3]
            fallback_bbox = [0, x, 0, y, 0, z]
            for i, col in enumerate(bbox_columns):
                df.at[idx, col] = fallback_bbox[i]
            print(f"Fallback bbox used for: {nii_path}")
        except Exception as e:
            print(f"Failed to fallback bbox for {nii_path}: {e}")
            missing_files.append(nii_path)

# Save the verified/cleaned CSV
df.to_csv(cleaned_csv, index=False)
print(f"✅ Saved verified CSV: {cleaned_csv}")
print(f"🔁 Bounding boxes updated for {updated_rows} entries")

# Optionally log missing files
if missing_files:
    log_path = "/gpfs/home/unalg01/jepa/src/datasets/missing_betmask_files.txt"
    with open(log_path, "w") as f:
        for path in missing_files:
            f.write(f"{path}\n")
    print(f"❌ Missing files logged at: {log_path}")
