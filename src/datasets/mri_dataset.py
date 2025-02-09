# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings
import math

from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import random

import torch
import torchio as tio

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
    in_chans=3,
    crop_size=224,
    random_clip_sampling=True,
    allow_clip_overlap=True, #False,
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
    training=True
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
        training=training)

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
        crop_size=224,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=True,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
        training=True
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

        # Load data from CSV
        samples, labels = [], []
        bbox = []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:

             if data_path[-4:] == '.csv':
                data = pd.read_csv(data_path)
                samples += data['nii_file_path'].tolist()
                labels += data['label'].tolist()
                
                                
                # Check for bounding box fields and add them if they exist
                if {'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'}.issubset(data.columns):
                    # Replace NaN values with a placeholder (e.g., -1) and convert to integers
                    bbox += data[['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']].fillna(-1).astype(int).values.tolist()
                else:
                    # If bounding box fields are missing, add placeholder values
                    bbox += [[-1, -1, -1, -1, -1, -1]] * len(data)

                # Count the number of samples in this dataset
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

    def __getitem__(self, index):
        sample = self.samples[index]
        # Label/annotations for video
        label = self.labels[index]
        bbox = self.bbox[index]  # Bounding box for this sample

        # GU_DEBUG: !!!!! bbox order does not match the volume dimensions
        #bbox = [bbox[4], bbox[5], bbox[0], bbox[1], bbox[2], bbox[3]]  # Shift order
        #bbox = [bbox[4], bbox[5], bbox[2], bbox[3], bbox[0], bbox[1]]  # Shift order

        # Load MRI volume
        volume = self.load_nifti_file(sample, bbox, self.in_chans)
         # debug_print
        # print(f'File name at {index}: ',sample)

        if volume is None:
            # Handle failed loading by skipping the sample
            warnings.warn(f'Failed to load volume at index {index}')
            return self.__getitem__((index + 1) % len(self.samples))

        if self.transform is not None:  #even if auto_augment is false, converts the volume to a tensor
            volume = self.transform(volume)
            
        #GU_ debug
        # affine = np.eye(4)
        # nifti_image = nib.Nifti1Image(volume.numpy(), affine)
        # nib.save(nifti_image, 'output_volume.nii')

        if not isinstance(volume, list): #for pretrain, volume is a tensor
            volume = self.intensity_normalize(volume)
            buffer, clip_indices = self.split_volume(volume)  # [T H W 1]
            buffer = buffer.permute(3, 0, 1, 2) # T H W C -> C T H W
            buffer = self.split_into_clips(buffer)
            #GU_debug
            # affine = np.eye(4)            
            # for i in range(self.num_clips): # Assuming buffer is a PyTorch tensor of shape [C, T, W, H]
            #     volume = buffer[i].squeeze(0)  # Remove the channel dimension (C)
            #     nifti_image = nib.Nifti1Image(volume.numpy(), affine)
            #     nib.save(nifti_image, f'buffer{i}_volume.nii')
            return buffer, label, clip_indices
        else: # for eval, volume is a list, this has to return a list of clips for clip aggregation in encoder to input to attentive pooler.
            volume = self.intensity_normalize(volume[0])
            buffer, clip_indices = self.split_volume(volume)  # [T H W 1]
            # buffer, clip_indices = self.split_volume(volume[0])  # [T H W 1]
            buffer = buffer.permute(3, 0, 1, 2) # T H W C -> C T H W
            buffer = self.split_into_clips(buffer)
            return [[clip] for clip in buffer], label, clip_indices
        
        # if self.transform is not None:
        #     buffer = [self.transform(clip) for clip in buffer]
        
        # plt.imsave('slice.png', buffer[3][0][8, :, :], cmap='gray') # Num_clips x Channel x T x H X W 
        

    def split_into_clips(self, video):
        """ Split video into a list of clips """
        fpc = self.frames_per_clip
        nc = self.num_clips
        return [video[:, i*fpc:(i+1)*fpc] for i in range(nc)]

    def load_nifti_file(self, file_path, bbox, in_chans=3):
        if not os.path.exists(file_path):
            warnings.warn(f'File not found: {file_path}')
            return None

        try:
            # Load the NIfTI file
            img = nib.load(file_path)
         
            # Convert to RAS+ orientation (ensures consistent L:R, A:P, B:U axes)
            #img = nib.as_closest_canonical(img)
            volume = img.get_fdata()

            #GU_Debug
            # affine = np.eye(4)
            # nifti_image = nib.Nifti1Image(volume, affine)
            # nib.save(nifti_image, 'Sorig_volume.nii')
            # print(f"Original Volume shape: {volume.shape}")

            # header = img.header
            # determine native orientation wrt smallest two spacing sizes
            # native = self.determine_native_orientation(header)
            #print(f"Native orientation wrt smallest spacings: {native}")

            if -1 in bbox: #check NaN or missing values
                bbox = [0, volume.shape[0], 0, volume.shape[1], 0, volume.shape[2]]  # Full volume bbox

            # Try to determine native in-plane orientation
            xsize, ysize, zsize = volume.shape
            dimensions = np.array([xsize, ysize, zsize])
            sorted_indices = np.argsort(dimensions)  # Sort dimensions (ascending)
            temporal_axis = sorted_indices[0]        # Smallest dimension -> temporal axis
            #inplane_axes = sorted_indices[1:]        # Two largest dimensions -> in-plane axes

            # crop the volume to the bounding box around the extracted brain bbox coordinates
            volume = self.crop_volume_bbox(volume, bbox, delta_box=6) #expand the bbox 
            
            #GU_ debug
            # affine = np.eye(4)
            # nifti_image = nib.Nifti1Image(volume, affine)
            # nib.save(nifti_image, 'Sinput_volume.nii')

            # GU_Debug: 
            # plt.imsave('slicemidAxial.png', volume[:,:,volume.shape[2]//2], cmap='gray')
            # plt.imsave('slicemidCoronal.png', volume[:,volume.shape[1]//2,:], cmap='gray')
            # plt.imsave('slicemidSagittal.png', volume[volume.shape[0]//2,:,:], cmap='gray') 
            # print(f"Volume shape before transpose: {volume.shape}")           
            

            select_random_orientation = self.training #If training is not true, i.e. in val loader, turn this off.
            if select_random_orientation:
                threshold_isotropy = 1.4 
                # For approximately isotropic volumes (where all dimensions are close), choose a random orientation
                if np.max(dimensions) / np.min(dimensions) < threshold_isotropy:  # Threshold for isotropy
                    orientations = ['axial', 'sagittal', 'coronal']
                    selected_orientation = random.choice(orientations)
                    if selected_orientation == 'axial':
                        volume = volume.transpose(2, 0, 1)  # Z, X, Y -> Axial: (Slices, H, W)
                    elif selected_orientation == 'sagittal':
                        volume = volume.transpose(0, 1, 2)  # X, Y, Z -> Sagittal: (Slices, H, W)
                    elif selected_orientation == 'coronal':
                        volume = volume.transpose(1, 0, 2)  # Y, X, Z -> Coronal: (Slices, H, W)
                else: # Assume that the native orientation is the axis along the smallest dimension
                    #if temporal_axis == 0:  # Sagittal
                    #    volume = volume.transpose(0, 1, 2)  # X, Y, Z -> Sagittal: (Slices, H, W)
                    if temporal_axis == 1:  
                        volume = volume.transpose(1, 0, 2)  # Y, X, Z -> (Slices, H, W)
                    elif temporal_axis == 2: 
                        volume = volume.transpose(2, 0, 1)  # Z, X, Y -> (Slices, H, W)
            else:
                if temporal_axis == 1:  
                        volume = volume.transpose(1, 0, 2)  # Y, X, Z -> (Slices, H, W)
                elif temporal_axis == 2: 
                        volume = volume.transpose(2, 0, 1)  # Z, X, Y -> (Slices, H, W)
            # print(f"Volume shape after transpose: {volume.shape}")

            # # Center crop each slice to a square (min dimension of in-plane axes)
            # h, w = volume.shape[1:3]  # Get in-plane dimensions (H, W)
            # min_dim = min(h, w)
            # start_h = (h - min_dim) // 2
            # start_w = (w - min_dim) // 2
            # volume = volume[:, start_h:start_h + min_dim, start_w:start_w + min_dim]  # Crop to [Slices, min_dim, min_dim]

            # Center crop each slice to a square with random offset for the larger dimension
            h, w = volume.shape[1:3]  # Get in-plane dimensions (H, W)
            if h != w:
                min_dim = min(h, w)
                if h > w:  # Height is larger
                    start_h = random.randint(0, h - min_dim)  # Random offset for height
                    start_w = 0  # Centered for width
                else:  # Width is larger
                    start_h = 0  # Centered for height
                    start_w = random.randint(0, w - min_dim)  # Random offset for width
                volume = volume[:, start_h:start_h + min_dim, start_w:start_w + min_dim]  # Crop to [Slices, min_dim, min_dim]

            # Clip intensities to a percentile range 
            volume = self.clip_intensity_percentile(volume, lower_percentile=1, upper_percentile=99)

            # Resize the in-plane dimensions to crop_size
            volume = self.resize(volume, crop_sizes={1: self.crop_size, 2: self.crop_size})

            # GU_Debug: save one png file for debugging
            # mid_slice_index = volume.shape[0]//2
            # plt.imsave('slicemid.png', volume[mid_slice_index], cmap='gray')
            # # GU_Debug: 
            # affine = np.eye(4)
            # nifti_image = nib.Nifti1Image(volume, affine)
            # nib.save(nifti_image, 'Soutput_volume.nii')
            
            # Preprocess the volume: intensity normalization
            volume = self.preprocess_volume(volume, in_chans)

            return volume

        except Exception as e:
            warnings.warn(f'Error loading {file_path}: {e}')
            return None

    import numpy as np

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

    # https://github.com/lunastra26/wmh-segmentation/blob/main/utils.py
    # following 3 functions are based on this git repo.
    def permuteOrientation(self, nii):
        target_dim = (256,256)
        img_dim = nii.header.get_data_shape()
        if img_dim[1] == target_dim[0] and img_dim[2] == target_dim[1]:
            img = np.fliplr(np.rot90(nii.get_data()))
        elif img_dim[0] == target_dim[0] and img_dim[0] == target_dim[1]:
            img = np.transpose(nii.get_data(),(2,0,1))
            img = np.rot90(img,-1)
        else:
            print('Permutation not supported: ', img_dim)
        return img

    def reformat_inputOrientation(self, ipImg,ipType,opShape):
        '''Creates axial, sagittal, and coronal reformatting of 3D FLAIR
        and crop 3D volume to size compatible with Orthogonal Nets
        These operations can be customized based on data orientation. 
        The following script assumes ipImg is oriented axially
        '''
        if ipType == 'Axial':
            opImg = ipImg    
        elif ipType == 'Sagittal':
            opImg = np.transpose(ipImg,(2,0,1))
        elif ipType == 'Coronal':
            opImg = np.transpose(ipImg,(2,1,0))         
        else:
            print('Data orientation not supported')
        print("Creating {} test volume for Orthogonal Net".format(ipType))
        origShape = opImg.shape 
        opImg = self.myCrop3D(opImg,opShape)
        return opImg, origShape 

    def myCrop3D(self, ipImg,opShape,padval=0):
        '''  Creates a 3D cropped volume from ipImg based on opShape (xDim,yDim)
        ipImg is a 3D volume    
        '''
        xDim,yDim = opShape
        zDim = ipImg.shape[2]
        if padval == 0:
            opImg = np.zeros((xDim,yDim,zDim))
        else:
            opImg = np.ones((xDim,yDim,zDim)) * np.min(ipImg)
        
        xPad = xDim - ipImg.shape[0]
        yPad = yDim - ipImg.shape[1]
        
        x_lwr = int(np.ceil(np.abs(xPad)/2))
        x_upr = int(np.floor(np.abs(xPad)/2))
        y_lwr = int(np.ceil(np.abs(yPad)/2))
        y_upr = int(np.floor(np.abs(yPad)/2))
        if xPad >= 0 and yPad >= 0:
            opImg[x_lwr:xDim - x_upr ,y_lwr:yDim - y_upr,:] = ipImg
        elif xPad < 0 and yPad < 0:
            xPad = np.abs(xPad)
            yPad = np.abs(yPad)
            opImg = ipImg[x_lwr: -x_upr ,y_lwr:- y_upr,:]
        elif xPad < 0 and yPad >= 0:
            xPad = np.abs(xPad)
            temp_opImg = ipImg[x_lwr: -x_upr,:,:]
            opImg[:,y_lwr:yDim - y_upr,:] = temp_opImg
        else:
            yPad = np.abs(yPad)
            temp_opImg = ipImg[:,y_lwr: -y_upr,:]
            opImg[x_lwr:xDim - x_upr,:,:] = temp_opImg
        return opImg

    
    def preprocess_volume(self, volume,in_chans=3):
       
        # Convert to float32
        volume = volume.astype(np.float32)

       # Expand the volume to have a channel dimension of size 1: [T, H, W, 1]
        volume = np.expand_dims(volume, -1)  #-1: increase last dim by 1

        # Replicate the volume along the last dimension to create 3 channels: [T, H, W, 3]
        if (in_chans > 1):
            volume = np.repeat(volume, in_chans, axis=-1)
        
        # Should output (T, H, W, 3)
        #print(f"Volume shape after preprocessing: {volume.shape}")  
        
        return volume
    
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
        
        resized_volume=np.empty([original_shape[0],crop_sizes[1],crop_sizes[2]])

        # Determine which axes to resize
        axes_to_resize = list(crop_sizes.keys())
        axes_to_resize.sort()  # Ensure consistent order

        # If resizing axes 1 and 2 (H and W), we can resize each 2D slice along axis 0
        if axes_to_resize == [1, 2]:
            D = original_shape[0]
            new_H = crop_sizes[1]
            new_W = crop_sizes[2]
            # resized_slices = []
            for i in range(D):
                # Extract the 2D slice
                slice_2d = volume[i, :, :]  # Shape: (H, W)
                # Convert to PIL Image
                slice_img = Image.fromarray(slice_2d)
                # Resize the image
                slice_resized = slice_img.resize((new_W, new_H), Image.BILINEAR)
                # Convert back to numpy array
                slice_resized = np.array(slice_resized)
                resized_volume[i, :, :] = slice_resized
                # resized_slices.append(slice_resized)
            # Stack the resized slices back into a 3D volume
            # volume = np.stack(resized_slices, axis=0)
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

        return resized_volume
        #return volume

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
