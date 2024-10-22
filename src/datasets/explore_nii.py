import nibabel as nib
from nibabel.orientations import aff2axcodes
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation

class RandomHorizontalFlip:
    def __init__(self, left_right_axis):
        self.left_right_axis = left_right_axis
    
    def __call__(self, volume):
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=self.left_right_axis).copy()
        return volume


# Standardizing Orientation to RAS
def reorient_to_RAS(img):
    # Get current orientation
    current_ornt = nib.io_orientation(img.affine)
    # Define desired orientation
    desired_ornt = axcodes2ornt(('R', 'A', 'S'))
    # Get the transform
    transform = ornt_transform(current_ornt, desired_ornt)
    # Apply the orientation
    data = img.get_fdata()
    reoriented_data = apply_orientation(data, transform)
    # Create new image with reoriented data
    new_affine = img.affine.copy()
    new_img = nib.Nifti1Image(reoriented_data, new_affine)
    return new_img

# Load the image
nii_file_name = '/media/yusuf/backup/ADNI-NC/ADNI_NC/016_S_4638/MT1__GradWarp__N3m/2013-02-01_10_55_25.0/I358089/ADNI_016_S_4638_MR_MT1__GradWarp__N3m_Br_20130206105826694_S181331_I358089.nii' 
img = nib.load(nii_file_name)
data = img.get_fdata()
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

# flip along axis 0 : width
axis_map = {0: axcodes[0], 1: axcodes[1], 2: axcodes[2]}
left_right_axis = [axis for axis, direction in axis_map.items() if direction in ('L', 'R')][0]

# Instantiate the transformation with the correct axis
transform = RandomHorizontalFlip(left_right_axis=left_right_axis)
