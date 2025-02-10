import nibabel as nib
import numpy as np
import pandas as pd
import nibabel.processing

# Function to calculate the bounding box of the brain mask
def calculate_bbox(mask):
    try:
        # Already loaded mask
        mask_data = mask.get_fdata()

        # Find the non-zero coordinates
        coords = np.argwhere(mask_data > 0)

        # Calculate the bounding box (min and max along each axis)
        if coords.size == 0:  # If the mask is empty
            return None
        bbox = {
            "xmin": coords[:, 0].min(),
            "xmax": coords[:, 0].max(),
            "ymin": coords[:, 1].min(),
            "ymax": coords[:, 1].max(),
            "zmin": coords[:, 2].min(),
            "zmax": coords[:, 2].max(),
        }
        return bbox
    except Exception as e:
        print("Error calculating bounding box!")
        return None

bet_mask_path = '/gpfs/data/sodicksonlab/gozde/ADNIall/ADNI/116_S_1232/MPR__GradWarp/2007-08-01_10_56_55.0/I72839/ADNI_116_S_1232_MR_MPR__GradWarp_Br_20070912180831275_S36796_I72839_betmask.nii.gz'
nii_path = '/gpfs/data/sodicksonlab/gozde/ADNIall/ADNI/116_S_1232/MPR__GradWarp/2007-08-01_10_56_55.0/I72839/ADNI_116_S_1232_MR_MPR__GradWarp_Br_20070912180831275_S36796_I72839.nii'

img = nib.load(nii_path)
mask = nib.load(bet_mask_path)

xsize, ysize, zsize = img.shape  # Get volume dimensions
print(f"size for volume is: {xsize,ysize,zsize}")

xsize, ysize, zsize = mask.shape  # Get volume dimensions
print(f"size for volume betmask is: {xsize,ysize,zsize}")

img_canonical = nib.as_closest_canonical(img)
mask_canonical = nib.as_closest_canonical(mask)

print("Canonical shape for NIfTI:", img_canonical.shape)
print("Canonical shape for BET mask:", mask_canonical.shape)

img_data = img.get_fdata()
header = img.header


print("Affine matrix of original NIfTI:")
print(img.affine)



# Resample the BET mask to match the original NIfTI orientation
# mask_resampled = nibabel.processing.resample_from_to(mask, img)

#xsize, ysize, zsize = mask_resampled.shape  # Get volume dimensions
#print(f"size for volume betmask is: {xsize,ysize,zsize}")

print("Affine matrix of BET mask NIfTI:")
print(mask.affine)


minI = np.min(img_data)
maxI = np.max(img_data)
print(f"min,max for volume is: {minI,maxI}")


# Calculate the bounding box
bbox = calculate_bbox(mask)
print(f"bbox calculated is: {bbox}")