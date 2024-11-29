import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import os

import nibabel as nib  # Import nibabel for NIfTI file handling

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

class MNIST3DDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        self.mnist = torchvision.datasets.MNIST(root=root, train=train,
                                                transform=transform,
                                                target_transform=target_transform,
                                                download=download)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        # Get the 2D image and label from the MNIST dataset
        image, label = self.mnist[idx]  # image is a PIL image

        # Convert image to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)  # Shape: [1, 28, 28]

        # Remove the channel dimension
        image = image.squeeze(0)  # Shape: [28, 28]

        # Create the 3D volume
        volume = self.create_3d_volume(image)  # Shape: [32, 28, 28]

        # Return the volume and label
        return volume, label

    def create_3d_volume(self, image_2d):
        # Number of slices
        total_slices = 32
        padding_slices = 5
        middle_slices = 22

        # Create empty volume
        volume = torch.zeros((total_slices, 28, 28))

        # Insert the replicated image into the middle slices
        volume[padding_slices:padding_slices+middle_slices] = image_2d

        return volume

    def save_volume_as_nii(self, volume, label, idx, save_dir='./nii_volumes'):
        # volume shape: [32, 28, 28]

        # Convert the volume to a NumPy array
        volume_np = volume.numpy().astype(np.float32)

        # Transpose the volume to match NIfTI format (X, Y, Z)
        # Original shape: (Slices, Height, Width)
        # Desired shape: (Width, Height, Slices)
        volume_np = np.transpose(volume_np, (2, 1, 0))  # Shape: [28, 28, 32]

        # GU_: Use this later if needed with flip Transpose and flip to correct orientation
        #volume_np = np.transpose(volume_np, (1, 2, 0))
        #volume_np = np.flip(volume_np, axis=0)

        # Create an affine transformation matrix
        affine = np.eye(4)
        # Optionally, set voxel sizes or orientations here

        # Create a NIfTI image
        nii_image = nib.Nifti1Image(volume_np, affine)

        # Create the directory to save NIfTI files if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save the NIfTI image
        save_path = os.path.join(save_dir, f'volume_{idx}_label_{label}.nii')
        nib.save(nii_image, save_path)
        print(f'Saved volume {idx} as {save_path}')

def visualize_sample(volume, label, idx, save_dir='./sample_images'):
    # volume shape: [32, 28, 28]
    # Create a grid of images
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(32):
        axes[i].imshow(volume[i], cmap='gray')
        axes[i].axis('off')
    plt.suptitle(f'Label: {label}')
    plt.tight_layout()

    # Create the directory to save images if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the figure as a PNG file
    save_path = os.path.join(save_dir, f'sample_{idx}_label_{label}.png')
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory
    print(f'Saved sample {idx} as {save_path}')

def main():
    # Define any transformations (if needed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add any other transformations here
    ])

    # Initialize the dataset
    mnist_3d_dataset = MNIST3DDataset(root='./data', train=True,
                                      transform=transform, download=True)

    # Create a DataLoader
    batch_size = 64
    data_loader = DataLoader(mnist_3d_dataset, batch_size=batch_size, shuffle=True)

    # Visualize and save a few samples
    for idx in range(5):
        volume, label = mnist_3d_dataset[idx]
        # Save the volume as a NIfTI file
        mnist_3d_dataset.save_volume_as_nii(volume, label, idx)
        # Optionally, visualize and save the sample images
        visualize_sample(volume, label, idx)

    # Optionally, save the entire transformed dataset as NIfTI files
    # Uncomment the following lines to save the dataset
   #  """
    for idx in range(len(mnist_3d_dataset)):
        volume, label = mnist_3d_dataset[idx]  # volume shape: [32, 28, 28]
        # Save the volume as a NIfTI file
        mnist_3d_dataset.save_volume_as_nii(volume, label, idx)
        if idx % 1000 == 0:
            print(f'Saved {idx} volumes.')
    # """

if __name__ == '__main__':
    main()
