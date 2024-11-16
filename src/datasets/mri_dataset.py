# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings

from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image

import torch

import sys 
sys.path.append('/home/gozde/medChangeDet/jepa')

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_mridataset(
    data_paths,
    batch_size,
    frames_per_clip=16, #8
    frame_step=1,
    num_clips=1,
    in_chans=3,
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
    duration=None,
    log_dir=None,
):
    dataset = MRIDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        in_chans=in_chans,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform)

    logger.info('MRIDataset dataset created')
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

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('MRIDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class MRIDataset(torch.utils.data.Dataset):
    """ MRI classification dataset. """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        frame_step=1,
        num_clips=1,
        in_chans=3,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.in_chans=in_chans
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.in_chans = in_chans

        # Load data from CSV
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:

             if data_path[-4:] == '.csv':
                data = pd.read_csv(data_path)
                samples += data['nii_file_path'].tolist()
                labels += data['label'].tolist()
    
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns
    
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample = self.samples[index]
        # Label/annotations for video
        label = self.labels[index]

        # Load MRI volume
        volume = self.load_nifti_file(sample,self.in_chans)
        if volume is None:
            # Handle failed loading by skipping the sample
            warnings.warn(f'Failed to load volume at index {index}')
            return self.__getitem__((index + 1) % len(self.samples))
      
        buffer, clip_indices = self.split_volume(volume)  # [T H W 1]
           
       
        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]
        
        return buffer, label, clip_indices

    def load_nifti_file(self, file_path,in_chans=3):
        if not os.path.exists(file_path):
            warnings.warn(f'File not found: {file_path}')
            return None

        try:
            # Load the NIfTI file
            img = nib.load(file_path)
            volume = img.get_fdata()

            # Transform from xyz (Sagittal) to zxy (Axial)
            volume = volume.transpose(2, 0, 1)  # Shape: (Z, X, Y)

    
            #volume = self.center_crop(volume, crop_sizes={1: 240, 2: 160})
            # Resize along axes 1 and 2 to size 224
            volume = self.resize(volume, crop_sizes={1: 224, 2: 224})
            
            # save one png file for debugging
            # plt.imsave('slice.png', volume[100], cmap='gray')

            # Preprocess the volume: intensity normalization
            volume, volume_mean, volume_std = self.preprocess_volume(volume,in_chans)
            
             # plt.imsave('slice_preprocessed.png', volume[100, :, :, 0]*volume_std + volume_mean, cmap='gray')

            return volume
    
        except Exception as e:
            warnings.warn(f'Error loading {file_path}: {e}')
            return None

    def center_crop(self, volume, crop_sizes):
        """
        Center crop the volume along specified axes to the desired sizes.

        Parameters:
            - volume (np.ndarray): The 3D MRI volume to be cropped.
            - crop_sizes (dict): A dictionary where keys are axis indices (0, 1, 2)
                         and values are the desired sizes along those axes.

        Returns:
            - volume (np.ndarray): The cropped volume.
        """
        shape = volume.shape  # Original shape after transpose
        slices = []
        for i in range(len(shape)):
            if i in crop_sizes:
                desired_size = crop_sizes[i]
                original_size = shape[i]
                if original_size < desired_size:
                    warnings.warn(f"Cannot crop axis {i} to size {desired_size} because it's smaller ({original_size}).")
                    start = 0
                    end = original_size
                else:
                    start = (original_size - desired_size) // 2
                    end = start + desired_size
                slices.append(slice(start, end))
            else:
                slices.append(slice(0, shape[i]))  # Use the full range for axes not being cropped
        volume = volume[tuple(slices)]
        return volume

    def preprocess_volume(self, volume,in_chans=3):
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


    def split_volume(self, volume):
        """  """
        fpc = self.frames_per_clip
        fstp = self.frame_step
        
        if self.duration is not None:
            try:
                fps = 1
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                warnings.warn(e)
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(volume) < clip_len:
            warnings.warn(f'skipping volume of length {len(volume)}')
            return [], None

        # Partition volume into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(volume) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(volume)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(volume) > clip_len:
                        clip_step = (len(volume) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        # buffer = vr.get_batch(all_indices).asnumpy()
        buffer = volume[all_indices]
        return buffer, clip_indices

    def resize(self, volume, crop_sizes):
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

        return volume

    def __len__(self):
        return len(self.samples)

    
if __name__ == "__main__":
    # Instantiate the RandomHorizontalFlip transformation
    transform = None

    # Set target_shape to (196, 256, 256)

    # Create the dataset and data loader
    dataset, data_loader, sampler = make_mridataset(
        data_paths='/home/gozde/medChangeDet/jepa/src/datasets/filtered_nii_small.csv',
        batch_size = 1,
        frames_per_clip=16,
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
        num_workers=1,
        pin_mem=True,
        duration=None,
        log_dir=None
    )

    for volumes, labels, patch_indices in data_loader:
        print(f'Volumes shape: {volumes.shape}')  # Shape: (batch_size * num_patches, 1, 64, 64, 64)
        print(f'Labels: {labels}')
        print(f'Patch indices: {patch_indices}')
        break
