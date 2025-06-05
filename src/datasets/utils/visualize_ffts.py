import nibabel as nib
import numpy as np
import torch
import torch.fft

# --- Load MRI Volume ---
nii_path = "/gpfs/home/unalg01/Downloads/ZReconstructed_mosaic_volume_c0-epoch10.nii.gz"  # reconstructed MRI volume
nii = nib.load(nii_path)
volume = nii.get_fdata()  # shape: [X, Y, Z]

# --- FFT: compute magnitude of 3D FFT ---
volume_tensor = torch.tensor(volume, dtype=torch.float32)

# Normalize (optional, to improve visibility of frequency patterns)
volume_tensor -= volume_tensor.mean()
volume_tensor /= volume_tensor.std() + 1e-6

# Compute FFT
fft_volume = torch.fft.fftn(volume_tensor, dim=(0, 1, 2))
fft_magnitude = torch.abs(fft_volume)

# Apply fftshift to move low frequencies to center
fft_shifted = torch.fft.fftshift(fft_magnitude, dim=(0, 1, 2))

# Optional: Log scale to improve visibility of high frequencies
fft_log = torch.log1p(fft_shifted)

# --- Save as NIfTI ---
fft_np = fft_log.cpu().numpy()
fft_nii = nib.Nifti1Image(fft_np, affine=nii.affine)
nib.save(fft_nii, "/gpfs/home/unalg01/Downloads/fft_shifted_logmag_epoch10.nii.gz")

print("Saved: fft_shifted_logmag.nii.gz")
