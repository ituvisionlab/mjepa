# mjepa: A 3D MRI self-supervised learning framework based on a modified V-JEPA
# Copyright (c) 2024–2025 [Gozde Unal, NYU]
#
# This file is created as inspired from the video_dataset from
# V-JEPA (https://github.com/facebookresearch/v-jepa)
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This codebase has been significantly modified for use in medical imaging and 3D MRI.
# All modifications are licensed under the original MIT license (or the applicable license).

import os
import pathlib
import warnings
import math
import time
from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage
import random

import torch
import torchio as tio

from collections import Counter
import sys 
sys.path.append('/gpfs/home/unalg01/jepa')

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_mridataset(
    data_paths,
    batch_size,
    frames_per_clip=16, #8
    frame_step=1,
    num_clips=1,
    in_chans=1,
    crop_size=224,
    random_clip_sampling=True,
    allow_clip_overlap=True, #False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    threshold_isotropy=1.4,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
    training=True,
    vol_type=None,
):
    dataset = MRIDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        in_chans=in_chans,
        crop_size=crop_size,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform,
        training=training,
        vol_type = vol_type,
        threshold_isotropy=threshold_isotropy,
        )

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
        collate_fn=collator, #MaskCollator #None used for eval, pytorch default batch collator
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
        in_chans=1,
        crop_size=224,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=True,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
        training=True,
        vol_type=None,
        threshold_isotropy=1.4  # for selecting a random view/orientation for near-isotropic volumes
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.in_chans=in_chans
        self.crop_size=crop_size
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.in_chans = in_chans
        self.training = training
        self.vol_type=vol_type
        self.threshold_isotropy = threshold_isotropy
        self.batchtime = 0 #for debugging time
        self.batchnum = 0 #for debugging time
        self.batchsize = 1 #for debugging time
        # self.loadTotaltime = 0 #for debugging time

        # Load data from CSV
        samples, labels = [], []
        bbox = []
        self.num_samples_per_dataset = []

        for data_path in self.data_paths: 
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
                labels += data['label'].tolist()

                if 'nii_file_path' in data.columns:
                    # Explicit single-channel scenario
                    samples += data['nii_file_path'].tolist()
                    self.contrast_names = ['single_channel']
                    self.in_chans = 1

                    single_bbox_cols = {'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'}
                    if single_bbox_cols.issubset(data.columns):
                        bbox += data[['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']].fillna(-1).astype(int).values.tolist()
                    else:
                        bbox += [[-1, -1, -1, -1, -1, -1]] * len(data)

                else:
                    # Multi-contrast case
                    contrast_cols = [col for col in data.columns if col.endswith('_path')][:self.in_chans]

                    if len(contrast_cols) == 0:
                        raise ValueError("No *_path columns found for multi-contrast MRI input.")

                    samples += list(zip(*[data[col] for col in contrast_cols]))
                    self.contrast_names = contrast_cols
                    self.in_chans = len(contrast_cols)

                    contrast_bbox_cols = {}
                    for cname in self.contrast_names:
                        prefix = cname.replace('_path', '')
                        expected_cols = [
                            f"{prefix}_xmin", f"{prefix}_xmax",
                            f"{prefix}_ymin", f"{prefix}_ymax",
                            f"{prefix}_zmin", f"{prefix}_zmax"
                        ]
                        if set(expected_cols).issubset(data.columns):
                            contrast_bbox_cols[prefix] = expected_cols

                    if contrast_bbox_cols:
                        bboxes_per_sample = []
                        for _, row in data.iterrows():
                            sample_bboxes = []
                            for prefix in contrast_bbox_cols:
                                cols = contrast_bbox_cols[prefix]
                                bbox_vals = pd.to_numeric(row[cols], errors='coerce').fillna(-1).astype(int).tolist()
                                sample_bboxes.append(bbox_vals)
                            bboxes_per_sample.append(sample_bboxes)
                        bbox += bboxes_per_sample
                    else:
                        bbox += [[-1, -1, -1, -1, -1, -1]] * len(data)

                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)


        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns
    
        # Store the loaded data
        self.samples = samples
        self.labels = labels
        self.bbox = bbox  # Store bounding boxes
        #DEBUG 
        class_counts = Counter(labels)
        total = sum(class_counts.values())
        logger.info(f"[DEBUG] DATA  Class Distribution:")
        for label, count in sorted(class_counts.items()):
            pct = 100 * count / total
            logger.info(f"  Class {label}: {count} samples ({pct:.2f}%)")


    def __getitem__(self, index):
        # data_start_time = time.time() # **Start timing data loading
        # self.batchnum +=1 # **Start timing data loading

        sample = self.samples[index]
        label = self.labels[index]
        bbox = self.bbox[index]  # Bounding box for this sample
       
        # Load MRI volume w/potentially missing contrasts (if in_chans > 1)
        #loadnii_start_time = time.time() # **Start timing nifti load
        volume, contrast_names_used = self.load_nifti_file(sample, bbox, self.in_chans) #T,H,W,C volume returns
        #loadnii_end_time = time.time() # **End timing nifti load
        C_max = self.in_chans #defines max # of allowed contrasts in multi-contrast case

        # GU_Debug_print
        # print(f'File name at {index}: ',sample)
               
        #GU_ debug     
        # print("Before transform, volume.shape:", volume.shape, type(volume))
        # for i in range(volume.shape[-1]):
        #     contrast = volume[..., i]  # shape: [T, H, W]
        #     # Save full volume as .nii
        #     nib.save(nib.Nifti1Image(contrast, affine=np.eye(4)), f"Zdebug_contrast_{i}.nii")
        #     #Save mid-slice as image
        #     mid_slice = contrast[contrast.shape[0] // 2, :, :]  # [H, W]
        #     plt.imsave(f"Zmidslice_contrast_{i}.png", mid_slice, cmap='gray')

        if volume is None:
            # Handle failed loading by skipping the sample
            warnings.warn(f'Failed to load volume at index {index}')
            return self.__getitem__((index + 1) % len(self.samples))

        # Determine available contrasts explicitly
        num_available_contrasts = volume.shape[-1]

        # Explicitly fill missing contrasts with zeros (padding)
        if num_available_contrasts < C_max:
            padded_volume = np.zeros(volume.shape[:-1] + (C_max,), dtype=volume.dtype) #T, H, W, C=max_contrasts
            padded_volume[..., :num_available_contrasts] = volume
            contrast_mask = np.array(
                [1] * num_available_contrasts + [0] * (C_max - num_available_contrasts),
                dtype=np.int64
            )
        else:
            padded_volume = volume
            contrast_mask = np.ones(C_max, dtype=np.int64)

        # Apply transforms explicitly
        if self.transform is not None:  #even if auto_augment is false, converts the volume to a tensor
            padded_volume = self.transform(padded_volume)  # for pretrain, transform returns a tensor, for eval returns list of T x H x W x C tensor

        #For debug: save transformed volume
        # for i in range(padded_volume.shape[-1]):
        #     contrast = padded_volume[..., i]  # shape: [T, H, W]
        #     #Save mid-slice as image
        #     mid_slice = contrast[contrast.shape[0] // 2, :, :]  # [H, W]
        #     plt.imsave(f"ZmidsliceXformed_contrast_{i}.png", mid_slice, cmap='gray')
        #     # Save full volume as .nii
        #     # Ensure contrast is a numpy array explicitly
        #     if isinstance(contrast, torch.Tensor):
        #         contrast_np = contrast.cpu().numpy()
        #     else:
        #         contrast_np = contrast
        #     nib.save(nib.Nifti1Image(contrast_np, affine=np.eye(4)), f"ZdebugXformed_contrast_{i}.nii")
           
        if not isinstance(padded_volume, list): #for pretrain, volume is already tensor
            volume = self.intensity_normalize(padded_volume)
            buffer, clip_indices = self.split_volume(volume)  # [T H W 1]
            #GU_ debug
            # nifti_image = nib.Nifti1Image(buffer.numpy(), affine = np.eye(4))
            # nib.save(nifti_image, 'Zbuffer.nii')
            buffer = buffer.permute(3, 0, 1, 2) # T H W C -> C T H W
            buffer = self.split_into_clips(buffer)

            # data_end_time = time.time()  # **End timing data load
            # data_gen_time = data_end_time - data_start_time # **End timing data
            # self.batchtime += data_gen_time #
            # if (self.batchnum == self.batchsize):
            #     logger.info(f"Batch Time: {self.batchtime:.4f} sec") # **Print timing of batch
            #     logger.info(f"LoadNifti Time: {loadnii_end_time-loadnii_start_time:.4f} sec") # **Print timing of load_nifti
            #     logger.info(f"Xform Time: {xform_end_time-xform_start_time:.4f} sec") # **Print timing of load_nifti
            #     self.batchnum = 0
            #     self.batchtime = 0 

            #GU_debug        
            # for i in range(self.num_clips): # Assuming buffer is a PyTorch tensor of shape [C, T, W, H]
            #     volume = buffer[i].squeeze(0)  # Remove the channel dimension (C)
            #     # nifti_image = nib.Nifti1Image(volume.numpy(),  affine = np.eye(4))
                # nib.save(nifti_image, f'Zclips{i}_volume.nii')
                #mid_slice_index = volume.shape[0] // 2  # Compute the middle slice index along the temporal axis
                # plt.imsave('ZmidSlice.png', volume[mid_slice_index, :, :].cpu().numpy(), cmap='gray')
            #END_GU_debug   
            return buffer, label, clip_indices, contrast_mask
        else: # for eval, volume is a list, this has to return a list of clips for clip aggregation in encoder to input to attentive pooler.
            if self.vol_type is None: #eval case
                volume = self.intensity_normalize(padded_volume[0]) #cancels volume list to a tensor T H W C
                buffer, clip_indices = self.split_volume(volume)  # [T H W 1]
                # buffer, clip_indices = self.split_volume(volume[0])  # [T H W 1]
                buffer = buffer.permute(3, 0, 1, 2) # T H W C -> C T H W
                buffer = self.split_into_clips(buffer)

                # GU_Debug: save each channel as a separate .nii file
                # tensor = buffer[0]  # since we only have one clip
                # for i in range(tensor.shape[0]):  # loop over channels
                #     contrast = tensor[i]  # shape: [T, H, W]
                #     nifti_image = nib.Nifti1Image(contrast.numpy(), affine=np.eye(4))
                #     nib.save(nifti_image, f'Zcontrast_{i}.nii')
                #end_debug
                return [buffer[0]], label, clip_indices, contrast_mask #return a buffer list of CxTxHxW
                #return [[clip] for clip in buffer], label, clip_indices, contrast_names_used  #Always a single clip in MRIDataset!!!!!!
            else: #DINO pretraining
                volume_out = []
                for i in range(len(padded_volume)):
                    buffer = self.intensity_normalize(padded_volume[i])
                    #buffer = self.intensity_normalize(volume[i])
                    buffer, clip_indices = self.split_volume(buffer)  # [T H W 1]
                    #GU_Debug
                    # nifti_image = nib.Nifti1Image(buffer.numpy(), affine=np.eye(4))
                    # nib.save(nifti_image, f'buffer_{i}.nii')
                    # end_debug
                    buffer = buffer.permute(3, 0, 1, 2) # T H W C -> C T H W
                    # buffer = self.split_into_clips(buffer)
                    volume_out.append(buffer) #volume_out is already a list of augmented views=clips
                return volume_out, label, clip_indices, contrast_mask
            
        # if self.transform is not None:
        #     buffer = [self.transform(clip) for clip in buffer]
        
        # plt.imsave('slice.png', buffer[3][0][8, :, :], cmap='gray') # Num_clips x Channel x T x H X W 
        

    def split_into_clips(self, video):
        """ Split video into a list of clips """
        fpc = self.frames_per_clip
        nc = self.num_clips
        return [video[:, i*fpc:(i+1)*fpc] for i in range(nc)]

    def load_nifti_file(self, file_path, bbox, in_chans=1):
        """
        Load single-contrast MRI volume, ensuring correct bbox usage.
        Returns:
            - volume: [T, H, W, 1] NumPy array
            - valid_contrast_names: ['single_channel']
        """
        vol = self._load_single_nifti(file_path, bbox)
        if vol is None:
            return None, []

        # Explicitly expand dimension to match [T, H, W, 1]
        vol = np.expand_dims(vol, axis=-1)
        vol = self.preprocess_volume(vol, 1)
        return vol, ['single_channel']

    # this function also handles multicontrast input
    # def load_nifti_file(self, file_path, bbox, in_chans=1):
    #     """
    #     Load MRI volumes (single or multi-contrast), ensuring correct bbox usage
    #     Returns:
    #         - volume: [T, H, W, C] (NumPy array): Returns a [T, H, W, C] volume for valid contrasts.
    #         - valid_contrast_names: list of str (subset of self.contrast_names or ['0'] if single-channel)
    #     """
    #     volume_list = []
    #     valid_names = []

    #     if in_chans == 1:
    #         vol = self._load_single_nifti(file_path, bbox)
    #         if vol is None:
    #             return None, []
    #         volume_list.append(vol)
    #         valid_names.append(self.contrast_names[0] if self.contrast_names else 'single_channel')
    #     else:
    #         for idx, path in enumerate(file_path):
    #             if not isinstance(path, str) or pd.isna(path):
    #                 continue  # Skip missing paths
    #             current_bbox = bbox[idx] if (isinstance(bbox, list) and isinstance(bbox[0], list)) else bbox
    #             vol = self._load_single_nifti(path, current_bbox)
    #             if vol is not None:
    #                 volume_list.append(vol)
    #                 valid_names.append(self.contrast_names[idx])

    #         if len(volume_list) == 0:
    #             warnings.warn(f"No valid contrasts found for file_path={file_path}")
    #             return None, []

    #     volume = np.stack(volume_list, axis=-1)  # [T, H, W, C]
    #     return self.preprocess_volume(volume, volume.shape[-1]), valid_names

    def _load_single_nifti(self, file_path, bbox):
        if not isinstance(file_path, str) or pd.isna(file_path):
            warnings.warn(f"Invalid or missing file path: {file_path}")
            return None

        if not os.path.exists(file_path):
            warnings.warn(f'File not found: {file_path}')
            return None

        try:
            #loadnii1_start_time = time.time() # **Start timing nifti loading 1
            img = nib.load(file_path) # H, W, T assumed
            volume = img.get_fdata()
            
            # #DEBUG
            # # Save full volume as .nii
            # nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), f"ZdebugLOADED.nii")
            # #Save mid-slice as image
            # mid_slice = volume[volume.shape[0] // 2, :, :]  # [H, W]
            # plt.imsave(f"ZmidsliceLOADED.png", mid_slice, cmap='gray')

            # Validate bbox
            if (not isinstance(bbox, list)) or (len(bbox) != 6) or (-1 in bbox):
                bbox = [0, volume.shape[0], 0, volume.shape[1], 0, volume.shape[2]]

            #loadnii1_end_time = time.time() # **

            volume = self.crop_volume_bbox(volume, bbox, delta_box=6)
            # enforce orientation
            volume = volume.transpose(2, 0, 1) # from [H, W, T] → [T, H, W]

            dims = np.array(volume.shape)
            temporal_axis = np.argsort(dims)[0]

            threshold_isotropy = self.threshold_isotropy
            if self.training and np.max(dims) / np.min(dims) < threshold_isotropy:
                orientation = random.choice(['axial', 'sagittal', 'coronal'])
                if orientation == 'axial':
                    volume = volume.transpose(2, 0, 1)
                elif orientation == 'sagittal':
                    volume = volume.transpose(0, 1, 2)
                elif orientation == 'coronal':
                    volume = volume.transpose(1, 0, 2)
            else:
                if temporal_axis == 1:
                    volume = volume.transpose(1, 0, 2)
                elif temporal_axis == 2:
                    volume = volume.transpose(2, 0, 1)

            # Square crop
            h, w = volume.shape[1:3]
            if h != w:
                min_dim = min(h, w)
                start_h = random.randint(0, h - min_dim) if h > w else 0
                start_w = random.randint(0, w - min_dim) if w > h else 0
                volume = volume[:, start_h:start_h + min_dim, start_w:start_w + min_dim]

            volume = self.clip_intensity_percentile(volume, lower_percentile=1, upper_percentile=99)
            volume = self.resize(volume, crop_sizes={1: self.crop_size, 2: self.crop_size}, target_slices=self.frames_per_clip)

            # logger.info(f"LoadNifti 1 Time: {loadnii1_end_time-loadnii1_start_time:.4f} sec") # **Print timing inside load_nifti
            return volume

        except Exception as e:
            warnings.warn(f'Error loading {file_path}: {e}')
            return None

    def crop_volume_bbox(self, volume, bbox, delta_box=5):
        """
        Crop the volume to the bounding box of the brain with a margin.

        Parameters:
        - volume: The 3D MRI volume (numpy array).
        - bbox: A list or tuple of bounding box coordinates [xmin, xmax, ymin, ymax, zmin, zmax].
        - delta_box: Optional margin to expand the bounding box (default: 5).
    
        Returns:
        - Cropped volume as a numpy array.
        """
        # Extract bbox coordinates
        xmin, xmax, ymin, ymax, zmin, zmax = bbox

        # Adjust bounding box with delta_box margin
        xmin = max(0, xmin - delta_box)
        ymin = max(0, ymin - delta_box)
        zmin = max(0, zmin - delta_box)
        xmax = min(volume.shape[0], xmax + delta_box)
        ymax = min(volume.shape[1], ymax + delta_box)
        zmax = min(volume.shape[2], zmax + delta_box)
        
        cropped_volume = volume[xmin:xmax, ymin:ymax, zmin:zmax]
        return cropped_volume


    def determine_axes(affine):
        """
        Determine the native orientation of the volume using the affine matrix.

        Parameters:
        - affine: Affine matrix from the NIfTI header.

        Returns:
        - slice_axis: The axis along which slices are acquired.
        - in_plane_axes: The remaining two axes that represent the in-plane orientation.
        """
        direction_cosines = affine[:3, :3]
        abs_cosines = np.abs(direction_cosines)
        axis_contributions = np.sum(abs_cosines, axis=0)
        sorted_axes = np.argsort(axis_contributions)
        slice_axis = sorted_axes[0]
        in_plane_axes = sorted_axes[1:]
        return slice_axis, in_plane_axes
        
    def preprocess_volume(self, volume, in_chans=1):
        volume = volume.astype(np.float32)

        if volume.ndim == 3:
            volume = np.expand_dims(volume, -1)  # [T, H, W, 1]
        elif volume.ndim == 4:
            assert volume.shape[-1] == in_chans

        # Only repeat if needed
        if volume.shape[-1] == 1 and in_chans > 1:
            volume = np.repeat(volume, in_chans, axis=-1)

        return volume

    #def preprocess_volume(self, volume,in_chans=1):
       
        # Convert to float32
        #volume = volume.astype(np.float32)

       # Expand the volume to have a channel dimension of size 1: [T, H, W, 1]
       # volume = np.expand_dims(volume, -1)  #-1: increase last dim by 1

        # Replicate the volume along the last dimension to create 3 channels: [T, H, W, 3]
        #if (in_chans > 1):
        #    volume = np.repeat(volume, in_chans, axis=-1)
        
        # Should output (T, H, W, 3)
        #print(f"Volume shape after preprocessing: {volume.shape}")        
        #return volume
    
    def intensity_normalize(self, volume):
       
       # Assuming 'volume' is a PyTorch tensor with shape [T, W, H, C]
        volume_mean = torch.mean(volume, dim=(0, 1, 2))  # Shape: [C]
        volume_std = torch.std(volume, dim=(0, 1, 2))  # Shape: [C]

        epsilon = 1e-8  # Small value to avoid division by zero

        # Set std to 1 if it's near zero (below a threshold)
        volume_std = torch.where(volume_std < epsilon, torch.tensor(1.0, dtype=volume_std.dtype, device=volume_std.device), volume_std)

        # Reshape mean and std for broadcasting
        volume_mean = volume_mean.view(1, 1, 1, -1)  # Shape: [1, 1, 1, C]
        volume_std = volume_std.view(1, 1, 1, -1)    # Shape: [1, 1, 1, C]

        # Normalize intensities
        volume = (volume - volume_mean) / (volume_std + epsilon)
        
        return volume
    
    def clip_intensity_percentile(self, volume, lower_percentile=1, upper_percentile=99):
        """
        Clip the intensity of the MRI volume to the specified percentile range.
        Parameters:
            volume (np.ndarray): The MRI volume to clip.
            lower_percentile (float): The lower percentile for clipping.
            upper_percentile (float): The upper percentile for clipping.
        Returns:
            np.ndarray: The intensity-clipped MRI volume.
        """
        lower_bound = np.percentile(volume, lower_percentile)
        upper_bound = np.percentile(volume, upper_percentile)
        return np.clip(volume, lower_bound, upper_bound)


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

        # if self.filter_short_videos and len(volume) < clip_len:
        #     warnings.warn(f'skipping volume of length {len(volume)}')
        #     return [], None
        # Repeat slices along the temporal dimension to match clip_len
        # if len(volume) < clip_len:
        #     # Convert to 0-based index
        #     interpolated_indices = torch.linspace(1, len(volume), steps=clip_len).round().long() - 1  

        #     # Clamp to ensure indices are within valid range
        #     interpolated_indices = interpolated_indices.clamp(0, len(volume)-1)

        #     # Use indexing to create the new tensor
        #     volume = volume[interpolated_indices]

            # print(volume.shape)  # Should be (clip_len, cropsize, cropsize)

        # Partition volume into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(volume) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len >= clip_len:
                # If partition_len >= clip len, then sample a random window of
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

    def resize(self, volume, crop_sizes, target_slices=None):
        """
        Resize volume along dim 1, 2 for in-plane resizing, and dim 0 for volume interpolation 
        Parameters:
            - volume (np.ndarray): Input 3D volume (D, H, W).
            - crop_sizes (dict): Desired spatial sizes along axes {1: H, 2: W}.
            - target_slices (int, optional): Desired number of slices along axis 0.
        Returns:
            - resized_volume (np.ndarray): Resized volume.
        """
        D, H, W = volume.shape

        # Determine scale factors for spatial resizing
        resize_scale_H = crop_sizes[1] / H
        resize_scale_W = crop_sizes[2] / W

        # Set depth (temporal) scale
        if target_slices:
            resize_scale_D = target_slices / D
        else:
            resize_scale_D = 1.0

        # Perform resizing along all three dimensions at once
        resized_volume = scipy.ndimage.zoom(
            volume,
            zoom=(resize_scale_D, resize_scale_H, resize_scale_W),
            order=1  # Linear interpolation
        )

        return resized_volume

    # def  resize(self, volume, crop_sizes, target_slices=None):
    #     """
    #     Resize the volume along specified axes to the desired sizes without using zoom.
    #     Additionally, interpolates the number of slices along the temporal axis (depth).

    #     Parameters:
    #     - volume (np.ndarray): The 3D MRI volume to be resized. Shape: (D, H, W)
    #     - crop_sizes (dict): A dictionary where keys are axis indices (0, 1, 2)
    #                         and values are the desired sizes along those axes.
    #     - target_slices (int, optional): The desired number of slices along the depth axis (D).

    #     Returns:
    #     - volume (np.ndarray): The resized volume.
    #     """
    #   # Get the original shape
    #     original_shape = volume.shape  # (D, H, W)
        
    #     resized_volume=np.empty([original_shape[0],crop_sizes[1],crop_sizes[2]])

    #     # Determine which axes to resize
    #     axes_to_resize = list(crop_sizes.keys())
    #     axes_to_resize.sort()  # Ensure consistent order

    #     # If resizing axes 1 and 2 (H and W), we can resize each 2D slice along axis 0
    #     if axes_to_resize == [1, 2]:
    #         D = original_shape[0]
    #         new_H = crop_sizes[1]
    #         new_W = crop_sizes[2]
    #         # resized_slices = []
    #         for i in range(D):
    #             # Extract the 2D slice
    #             slice_2d = volume[i, :, :]  # Shape: (H, W)
    #             # Convert to PIL Image
    #             slice_img = Image.fromarray(slice_2d)
    #             # Resize the image
    #             slice_resized = slice_img.resize((new_W, new_H), Image.BILINEAR)
    #             # Convert back to numpy array
    #             slice_resized = np.array(slice_resized)
    #             resized_volume[i, :, :] = slice_resized
    #             # resized_slices.append(slice_resized)
    #         # Stack the resized slices back into a 3D volume
    #         # volume = np.stack(resized_slices, axis=0)
    #     else:
    #         raise NotImplementedError("Resizing along axes other than 1 and 2 is not implemented.")

    #      # **Step 2: Temporal Interpolation along axis=0
    #     if target_slices and target_slices != resized_volume.shape[0]: #and target_slices < resized_volume.shape[0]:
    #         depth_scale = target_slices / resized_volume.shape[0]  # Compute scaling factor along D
    #         resized_volume = scipy.ndimage.zoom(resized_volume, zoom=(depth_scale, 1, 1), order=1)  # Linear interpolation

    #     # debug_save
    #     # output_filename = "zVolume_resized_output.nii.gz"
    #     # volout = np.squeeze(resized_volume)
    #     # nii_img = nib.Nifti1Image(volout, affine=np.eye(4))
    #     # nib.save(nii_img, output_filename)

    #     return resized_volume
    #     #return volume

    def __len__(self):
        return len(self.samples)
    
    def filter_nifti(self, img, min_fov=50, max_spacing=6.5):
       
        header = img.header
    
        # Extract voxel dimensions and image dimensions
        voxel_spacing = header.get_zooms()[:3]  # pixdim[1:3]
        image_dimensions = header.get_data_shape()[:3]  # dim[1:3]
    
        # Compute field of view for each axis
        fov = [spacing * dim for spacing, dim in zip(voxel_spacing, image_dimensions)]
    
        # Apply filtering criteria
        if any(f < min_fov for f in fov) or any(s > max_spacing for s in voxel_spacing):
            return False  # Exclude this file
        return True  # Include this file
    
    def resample_image(self, img, target=None):
        # Define the resample transform with target spacing of [1x1x1] mm
        if target == None:
            target=(1, 1, 1)
        
        resample = tio.Resample(target=target)

        # Apply the resample transform
        resampled_image = resample(img)

        # Save the resampled image if needed
        # resampled_image.save('resampled_image.nii')
        
        return resampled_image

    def determine_native_orientation(self,header): 
        """
        Determine the native acquisition orientation based on voxel spacing.

        Args:
            header: The NIfTI header containing voxel spacing information.

        Returns:
            str: The native orientation ('axial', 'sagittal', 'coronal').
        """
        # Extract voxel spacing (assumes pixdim[1:4] corresponds to x, y, z spacing)
        spacings = header.get_zooms()[:3]  # x, y, z spacing

        # Sort the spacings to identify in-plane and through-plane axes
        sorted_indices = sorted(range(len(spacings)), key=lambda i: spacings[i])

        # Identify the through-plane axis (largest spacing)
        slice_axis = sorted_indices[-1]

        # Determine the orientation based on in-plane axes
        in_plane_axes = sorted_indices[:2]  # Smallest two spacings

        if set(in_plane_axes) == {0, 1}:  # X and Y are in-plane
            return 'axial'
        elif set(in_plane_axes) == {1, 2}:  # Y and Z are in-plane
            return 'sagittal'
        elif set(in_plane_axes) == {0, 2}:  # X and Z are in-plane
            return 'coronal'
        else:
            raise ValueError("Unable to determine orientation based on spacing.")

if __name__ == "__main__":
    # Instantiate the RandomHorizontalFlip transformation
    transform = None

    # Set target_shape to (196, 256, 256)

    # Create the dataset and data loader
    dataset, data_loader, sampler = make_mridataset(
        data_paths='/gpfs/home/unalg01/jepa/src/datasets/filtered_nii_small.csv',
        batch_size = 1,
        frames_per_clip=16,
        frame_step=1,
        num_clips=1,
        random_clip_sampling=True,
        allow_clip_overlap=True,
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
