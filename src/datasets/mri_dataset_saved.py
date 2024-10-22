import os
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from logging import getLogger
from scipy import ndimage
import itertools

import sys 
sys.path.append('/home/gozde/medChangeDet/jepa')

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

logger = getLogger()


def make_mridataset(
    data_paths,
    batch_size,
    frames_per_clip=196,
    frame_step=1,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    num_patches=1,
    random_patch_sampling=True,
    allow_patch_overlap=False,
    mask_ratio=0.50,
    duration=None,
    log_dir=None,
):
    patch_size = 16
    num_frames = patch_size[0]
    dataset = MRIDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        transform=transform,
        shared_transform=shared_transform,
        rank=rank,
        world_size=world_size,
        drop_last=drop_last,
        target_shape=target_shape,
        patch_size=patch_size,
        patch_step=patch_step,
        num_patches=num_patches,
        num_frames = num_frames,
        random_patch_sampling=random_patch_sampling,
        allow_patch_overlap=allow_patch_overlap,
        mask_ratio=mask_ratio,
    )

    logger.info('MRIDataset created')

    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    if collator is None:
        collator = default_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)

    logger.info('MRIDataset data loader created')

    return dataset, data_loader, dist_sampler

class RandomHorizontalFlip:
    def __call__(self, volume):
        if np.random.rand() > 0.5:
            # Flip along Axis 0 (Width / Left-Right)
            volume = np.flip(volume, axis=0).copy()
        return volume


class MRIDataset(Dataset):
    def __init__(
        self,
        data_paths,
        transform=None,
        shared_transform=None,
        datasets_weights=None,
        rank=0,
        world_size=1,
        drop_last=True,
        target_shape=(196, 256, 256),  # Width, Depth, Height
        patch_size=(32, 32, 32),
        patch_step=(32, 32, 32),
        num_patches=1,
        random_patch_sampling=True,
        allow_patch_overlap=False,
        mask_ratio=0.15,
    ):
        self.data_paths = data_paths
        self.transform = transform
        self.shared_transform = shared_transform
        self.datasets_weights = datasets_weights
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.target_shape = target_shape
        self.patch_size = patch_size
        self.patch_step = patch_step
        self.num_patches = num_patches
        self.random_patch_sampling = random_patch_sampling
        self.allow_patch_overlap = allow_patch_overlap
        self.mask_ratio = mask_ratio

        # Load data from CSV
        self.samples, self.labels = self.load_samples_from_csv(self.data_paths)
        self.num_samples_per_dataset = [len(self.samples)]

        # Handle sample weights if necessary (optional for multi datasets)
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = [self.datasets_weights[label] for label in self.labels]

    def load_samples_from_csv(self, data_paths):
        data = pd.read_csv(data_paths)
        samples = data['nii_file_path'].tolist()
        labels = data['label'].tolist()
        return samples, labels

    def __getitem__(self, index):
        sample_path = self.samples[index]
        label = self.labels[index]

        # Load MRI volume
        volume = self.load_nifti_file(sample_path)
        if volume is None:
            # Handle failed loading by skipping the sample
            warnings.warn(f'Failed to load volume at index {index}')
            return self.__getitem__((index + 1) % len(self.samples))

        # Apply shared transforms if any
        if self.shared_transform is not None:
            volume = self.shared_transform(volume)

        # Extract patches
        patches, patch_indices = self.extract_patches(volume)

        # Apply masking
        patches = self.apply_masking(patches)

        # Apply individual transforms to each patch
        if self.transform is not None:
            patches = [self.transform(patch) for patch in patches]

        return patches, label, patch_indices

    def load_nifti_file(self, file_path):
        if not os.path.exists(file_path):
            warnings.warn(f'File not found: {file_path}')
            return None

        try:
            img = nib.load(file_path)
            volume = img.get_fdata()
            # Preprocess the volume
            volume = self.preprocess_volume(volume)
            return volume
        except Exception as e:
            warnings.warn(f'Error loading {file_path}: {e}')
            return None

    def preprocess_volume(self, volume):
        # Resize the volume to the target shape
        volume = self.resize_volume(volume, self.target_shape)
        # Normalize intensities
        volume = (volume - np.mean(volume)) / np.std(volume)
        # Convert to float32
        volume = volume.astype(np.float32)
        return volume

    def resize_volume(self, volume, target_shape):
        # Calculate the zoom factors for each axis
        factors = (
            target_shape[0] / volume.shape[0],  # Width scaling factor (Axis 0)
            target_shape[1] / volume.shape[1],  # Depth scaling factor (Axis 1)
            target_shape[2] / volume.shape[2],  # Height scaling factor (Axis 2)
        )
        # Use scipy.ndimage.zoom to resize
        volume = ndimage.zoom(volume, factors, order=1)  # order=1 for linear interpolation
        return volume

    def extract_patches(self, volume):
        """
        Extract patches from the 3D volume.
        """
        D, H, W = volume.shape
        p_d, p_h, p_w = self.patch_size
        s_d, s_h, s_w = self.patch_step

        # Generate patch starting indices
        d_indices = range(0, D - p_d + 1, s_d)
        h_indices = range(0, H - p_h + 1, s_h)
        w_indices = range(0, W - p_w + 1, s_w)

        if self.random_patch_sampling:
            patches = []
            patch_indices = []
            for _ in range(self.num_patches):
                idx_d = np.random.choice(d_indices)
                idx_h = np.random.choice(h_indices)
                idx_w = np.random.choice(w_indices)
                patch = volume[
                    idx_d: idx_d + p_d,
                    idx_h: idx_h + p_h,
                    idx_w: idx_w + p_w,
                ]
                patches.append(patch)
                patch_indices.append((idx_d, idx_h, idx_w))
        else:
            all_indices = list(itertools.product(d_indices, h_indices, w_indices))
            if not self.allow_patch_overlap:
                step = max(1, len(all_indices) // self.num_patches)
                selected_indices = all_indices[::step][:self.num_patches]
            else:
                selected_indices = all_indices[:self.num_patches]
            patches = []
            patch_indices = []
            for idx_d, idx_h, idx_w in selected_indices:
                patch = volume[
                    idx_d: idx_d + p_d,
                    idx_h: idx_h + p_h,
                    idx_w: idx_w + p_w,
                ]
                patches.append(patch)
                patch_indices.append((idx_d, idx_h, idx_w))

        return patches, patch_indices

    def apply_masking(self, patches):
        """
        Apply masking to the patches.
        """
        num_patches = len(patches)
        num_mask = int(self.mask_ratio * num_patches)
        if num_mask > 0:
            mask_indices = np.random.choice(num_patches, num_mask, replace=False)
            for idx in mask_indices:
                patches[idx] = np.zeros_like(patches[idx])  # Or replace with noise
        return patches

    def __len__(self):
        return len(self.samples)


def default_collate_fn(batch):
    # Unpack the batch
    batch_patches, batch_labels, batch_patch_indices = zip(*batch)
    all_patches = []
    all_patch_indices = []
    labels = []
    for patches, label, patch_indices in zip(batch_patches, batch_labels, batch_patch_indices):
        all_patches.extend(patches)
        all_patch_indices.extend(patch_indices)
        labels.extend([label] * len(patches))
    # Stack patches into a tensor
    volumes = np.stack([p[np.newaxis, ...] for p in all_patches])  # Shape: (total_patches, 1, D, H, W)
    volumes = torch.from_numpy(volumes)
    labels = torch.tensor(labels, dtype=torch.long)
    return volumes, labels, all_patch_indices




if __name__ == "__main__":
    # Instantiate the RandomHorizontalFlip transformation
    transform = RandomHorizontalFlip()

    # Set target_shape to (196, 256, 256)
    target_shape = (196, 256, 256)

    # Create the dataset and data loader
    dataset, data_loader, sampler = make_mridataset(
        data_paths='/home/gozde/medChangeDet/jepa/src/datasets/filtered_nii_small.csv',
        batch_size=2,
        transform=transform,
        num_workers=4,
        target_shape=target_shape,
        patch_size=(64, 64, 64),
        patch_step=(32, 32, 32),
        num_patches=8,
        random_patch_sampling=True,
        allow_patch_overlap=False,
        mask_ratio=0.50,
    )

    for volumes, labels, patch_indices in data_loader:
        print(f'Volumes shape: {volumes.shape}')  # Shape: (batch_size * num_patches, 1, 64, 64, 64)
        print(f'Labels: {labels}')
        print(f'Patch indices: {patch_indices}')
        break
