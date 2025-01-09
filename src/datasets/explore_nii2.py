import nibabel as nib
from nibabel.orientations import aff2axcodes
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from PIL import Image
import torchio as tio

# Standardizing Orientation to RAS
def reorient_to_RAS(img):
    # Get current orientation
    current_ornt = nib.io_orientation(img.affine)

    axcodes = aff2axcodes(img.affine)
    print(f'Current Axial codes: {axcodes}')
    print(img.affine)

    # Define desired orientation
    desired_ornt = axcodes2ornt(('R', 'A', 'S'))
    # Get the transform
    transform = ornt_transform(current_ornt, desired_ornt)
    # Apply the orientation
    data = img.get_fdata()
    reoriented_data = apply_orientation(data, transform)
   
    # Create new image with reoriented data
    new_affine = img.affine.copy()
    
    axcodes = aff2axcodes(new_affine)
    print(f'Reoriented Axial codes: {axcodes}')

    #new_img = nib.Nifti1Image(reoriented_data, new_affine)
    new_img = nib.Nifti1Image(reoriented_data, affine=np.eye(4))
    
    return new_img

def resize(volume, crop_sizes):
        """
        Resize the volume along specified axes to the desired sizes without using zoom.

        Parameters:
        - volume (np.ndarray): The 3D MRI volume to be resized.
        - crop_sizes (dict): A dictionary where keys are axis indices (0, 1, 2)
                            and values are the desired sizes along those axes.

        Returns:
        - volume (np.ndarray): The resized volume.
        """
        # Get the original shape
        original_shape = volume.shape  # (D, H, W)
        
        # Determine which axes to resize
        axes_to_resize = list(crop_sizes.keys())
        axes_to_resize.sort()  # Ensure consistent order

        # If resizing axes 1 and 2 (H and W), we can resize each 2D slice along axis 0
        if axes_to_resize == [1, 2]:
            D = original_shape[0]
            new_H = crop_sizes[1]
            new_W = crop_sizes[2]
            resized_slices = []
            for i in range(D):
                # Extract the 2D slice
                slice_2d = volume[i, :, :]  # Shape: (H, W)
                # Convert to PIL Image
                slice_img = Image.fromarray(slice_2d)
                # Resize the image
                slice_resized = slice_img.resize((new_W, new_H), Image.BILINEAR)
                # Convert back to numpy array
                slice_resized = np.array(slice_resized)
                resized_slices.append(slice_resized)
            # Stack the resized slices back into a 3D volume
            volume = np.stack(resized_slices, axis=0)
        else:
            raise NotImplementedError("Resizing along axes other than 1 and 2 is not implemented.")

        # debug_save
        # output_dir = "volume_resized_output"
        # os.makedirs(output_dir, exist_ok=True)
        # output_path = os.path.join(output_dir, "volume_resized_output.nii.gz")
        # volout = np.squeeze(volume)
        # volout = np.transpose(volout, (0, 2, 1))
        # nii_img = nib.Nifti1Image(volout, affine=np.eye(4))
        # nib.save(nii_img, output_path)

        return volume

def preprocess_volume(volume,in_chans=3):
        volume_mean = np.mean(volume)
        volume_std = np.std(volume)
        # Normalize intensities
        volume = (volume - volume_mean) / volume_std
        # Convert to float32
        volume = volume.astype(np.float32)

       # Expand the volume to have a channel dimension of size 1: [T, H, W, 1]
        volume = np.expand_dims(volume, -1)  #-1: increase last dim by 1

        # Replicate the volume along the last dimension to create 3 channels: [T, H, W, 3]

        if (in_chans > 1):
            volume = np.repeat(volume, in_chans, axis=-1)
    
        # Should output (T, H, W, 3)
        #print(f"Volume shape after preprocessing: {volume.shape}")  
        
        return volume, volume_mean, volume_std

def resample_to_target_spacing(nifti_image, target_spacing=(1, 1, 1), output_path=None):
    data = nifti_image.get_fdata()
    affine = nifti_image.affine
    current_spacing = nifti_image.header.get_zooms()[:3]
    
    # Create a TorchIO Image
    tio_image = tio.ScalarImage(tensor=data[np.newaxis], affine=affine)
    
    # Define the resample transform
    resample = tio.Resample(target=target_spacing)
    
    # Apply the resample transform
    resampled_tio_image = resample(tio_image)
    
    # Extract the resampled data and affine
    resampled_data = resampled_tio_image.tensor.numpy().squeeze(0)
    resampled_affine = resampled_tio_image.affine
    
    # Save the resampled image if output path is provided
    if output_path:
        resampled_nifti = nib.Nifti1Image(resampled_data, resampled_affine)
        nib.save(resampled_nifti, output_path)
    
    return resampled_data, resampled_affine

# Load the image
nii_file_name = '/media/yusuf/disk1/ADNI/OAS30803_MR_d4473/anat2/sub-OAS30803_ses-d4473_acq-TSE_T2w.nii.gz'
#nii_file_name = '/media/yusuf/disk1/ADNI/OAS31092_MR_d0203/anat2/sub-OAS31092_ses-d0203_acq-TSE_run-02_T2w.nii.gz'
img = nib.load(nii_file_name)
data = img.get_fdata()

output_path = 'resampled_image.nii'
resampled_data, resampled_affine = resample_to_target_spacing(img, target_spacing=(1, 1, 1), output_path=output_path)


affine = img.affine

# Get the axial codes
axcodes = aff2axcodes(affine)
print(f'Axial codes: {axcodes}')

# Map axes to anatomical directions
axis_map = {
    0: axcodes[0],  # Axis 0 direction
    1: axcodes[1],  # Axis 1 direction
    2: axcodes[2],  # Axis 2 direction
}

print(f'Axis mapping: {axis_map}')

# Determine which axis corresponds to Left-Right
for axis, direction in axis_map.items():
    if direction in ('L', 'R'):
        left_right_axis = axis
    elif direction in ('P', 'A'):
        anterior_posterior_axis = axis
    elif direction in ('I', 'S'):
        inferior_superior_axis = axis

print(f'Left-Right axis: {left_right_axis}')
print(f'Anterior-Posterior axis: {anterior_posterior_axis}')
print(f'Inferior-Superior axis: {inferior_superior_axis}')

# re-orient all your images to the same orientation
# img = nib.load(nii_file_name)
reoriented_img = reorient_to_RAS(img)
data = reoriented_img.get_fdata()


#Do a CANONICAL transform to bring the MRI volume to RAS+ orientation: L:R, A:P, Bottom-Up
#img = nib.funcs.as_closest_canonical(img)

volume = data
     
# Transform from xyz to zxy (to make it like video)
volume = volume.transpose(2, 0, 1)  # Shape: (Z, X, Y)
crop_size=224
volume = resize(volume, crop_sizes={1: crop_size, 2: crop_size})
            
plt.imsave('sliceOut.png', volume[22], cmap='gray')

# Preprocess the volume: intensity normalization
volume, volume_mean, volume_std = preprocess_volume(volume,in_chans=1)
            
# GU_Debug:
plt.imsave('slice_preprocessed.png', volume[22, :, :, 0]*volume_std + volume_mean, cmap='gray')
