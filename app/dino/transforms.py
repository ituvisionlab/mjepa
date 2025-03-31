# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torchvision.transforms as transforms
import time
import torchio as tio
import numpy as np
import nibabel as nib

# from monai.transforms import (
#     Compose,
#     RandAffine,
#     RandFlip,
#     Rand3DElastic,
#     EnsureChannelFirst,
#     ToTensorD
# )

import random
import src.datasets.utils.video.transforms as video_transforms
from src.datasets.utils.video.randerase import RandomErasing

import matplotlib.pyplot as plt


# Function to create global and local transformations for DINO

def make_dino_transforms(
    num_global_views=2,
    num_local_views=6,
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(1.0,1.0), #(3/4, 4/3),
    random_resize_scale=(0.2, 1.0),
    rot_degree = 15.0,
    auto_augment=False,
    crop_size= 224,
    local_crop_ratio=0.8,
    max_offset_fraction=0.1,
    intensity_gamma=0.2,
    random_bias=0.2,
    random_noise=0.025,
    #normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    normalize=((0.0),(1))
):
    _volume_augmentation = MRITransform(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        auto_augment=auto_augment,
        crop_size=crop_size,
        local_crop_ratio=local_crop_ratio,
        max_offset_fraction=max_offset_fraction,
        num_global_views=num_global_views,
        num_local_views=num_local_views,
        random_resize_scale=random_resize_scale,
        rot_degree = rot_degree,
        intensity_gamma=intensity_gamma,
        random_bias=random_bias,
        random_noise=random_noise
    )
    return _volume_augmentation
    

class MRITransform(object):

    def __init__(
        self,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(1.0,1.0), #(0.9, 1.1),
        random_resize_scale=(0.8,1.0), # use for crop_retention ratio
        rot_degree = 0.0, 
        auto_augment=False,
        crop_size=224,
        local_crop_ratio=0.8,
        max_offset_fraction=0.1,
        num_global_views=2,
        num_local_views=6,
        intensity_gamma=0.2,
        random_bias=0.2,
        random_noise=0.025,
    ):
        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.rot_degree = rot_degree
        self.auto_augment = auto_augment
        self.crop_size = crop_size
        self.local_crop_ratio = local_crop_ratio
        self.max_offset_fraction=max_offset_fraction
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.crop_retention=random_resize_scale[0]
        self.intensity_gamma=intensity_gamma
        self.random_bias=random_bias
        self.random_noise=random_noise
  
    def __call__(self, buffer):

        buffer = torch.tensor(buffer, dtype=torch.float32)
        # buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W
        
        # For DINO: augmentation is always executed
        #if self.auto_augment:
        # Permute to shape C H W T for TorchIO compatibility
        buffer = buffer.permute(3, 1, 2, 0)  # T H W C -> C H W T


        global_transforms = transforms.Compose([
            self.get_global_spatial_transforms(),
            self.get_intensity_transforms(),
            #transforms.ToTensor(),
        ])

        local_transforms = transforms.Compose([
            self.get_spatial_transforms(),
            self.get_intensity_transforms(),
            #transforms.ToTensor(),
        ])

        buffer_global=[]
        buffer_local =[]
       # Apply the transforms
        for i in range(self.num_global_views):
            buffer_g = global_transforms(buffer)
            buffer_g = buffer_g.permute(3, 1, 2, 0) #permute back:C H W T ->  T H W C
            buffer_global.append(buffer_g)

        image = tio.Image(tensor=buffer, type=tio.INTENSITY) #needed for local xforms that use random_crop fn
        subject = tio.Subject(image=image)  # <- Always wrap
        for i in range(self.num_local_views):
            buffer_l = local_transforms(subject).image.tensor #extract tensor
            buffer_l = buffer_l.permute(3, 1, 2, 0) #permute back:C H W T ->  T H W C
            buffer_local.append(buffer_l)
    
        #GU_ debug
        # mid_slice_index = buffer_global.shape[0] // 2  # Compute the middle slice index along the temporal axis
        # plt.imsave('zxformedBufferG.png', buffer[mid_slice_index, :, :,0].cpu().numpy(), cmap='gray')
        # mid_slice_index = buffer_local.shape[0] // 2  # Compute the middle slice index along the temporal axis
        # plt.imsave('zxformedBufferL.png', buffer[mid_slice_index, :, :,0].cpu().numpy(), cmap='gray')
        # #GU_ debug
        # affine = np.eye(4)
        # nifti_image = nib.Nifti1Image(buffer_global.numpy(), affine)
        # nib.save(nifti_image, 'zxformed_volume_global.nii')
        # nifti_image = nib.Nifti1Image(buffer_local.numpy(), affine)
        # nib.save(nifti_image, 'zxformed_volume_local.nii')
        # Concatenate all transformed buffers in the desired order

        buffer_overall = buffer_global + buffer_local  # First two are global, last four are local
        return buffer_overall  # Return as a list of 6 transformed volumes
    
 
    def get_global_spatial_transforms(self):
        """Creates a set of spatial transformations."""       
        spatial_transforms = {
            tio.RandomAffine(scales=self.random_resize_aspect_ratio, degrees=self.rot_degree),  # Random affine
            tio.RandomFlip(axes=(0,)),  # Flip along left-right
            tio.Lambda(lambda x: x),  # Identity transform
        }
        return tio.OneOf(spatial_transforms)
    
        #if we want to give weights to different transforms
        # return tio.OneOf({
            # tio.RandomAffine(...): 0.4,
            # tio.RandomFlip(...): 0.4,
            # tio.Lambda(lambda x: x): 0.2  # 20% chance to do nothing
        #})

    def get_spatial_transforms(self):
        """Creates a set of spatial transformations."""       
        spatial_transforms = {
            tio.RandomAffine(scales=self.random_resize_aspect_ratio, degrees=self.rot_degree),  # Random affine
            tio.RandomFlip(axes=(0,)),  # Flip along left-right
            self.random_center_crop(),
        }
        return tio.OneOf(spatial_transforms)


    def get_intensity_transforms(self):
        """Creates a set of intensity transformations."""
        intensity_transforms = {
            tio.RandomGamma(log_gamma=(-self.intensity_gamma, self.intensity_gamma)),
            tio.RandomNoise(mean=0.0, std=self.random_noise),
            tio.Lambda(lambda x: x),  # Identity transform
        }
        return tio.OneOf(intensity_transforms)

#-----Only local transforms use this
    def random_center_crop(self):
        """
        Creates a custom center crop transform that retains local_crop_ratio of the input size
        and allows random offsets. Then resizes back to original size.
        """
        class CenterCropAndResizeTransform(tio.Transform):
            def __init__(self, local_crop_ratio, max_offset_fraction):
                super().__init__()
                self.local_crop_ratio = local_crop_ratio
                self.max_offset_fraction = max_offset_fraction

            def apply_transform(self, subject: tio.Subject) -> tio.Image:
                image = subject.get_first_image()  # Assumes there's only one image
                tensor = image.tensor  # (C, D, H, W)
                affine = image.affine
                _, D, H, W = tensor.shape

                # Determine random crop size
                crop_D = int(D * np.random.uniform(self.local_crop_ratio, 1.0))
                crop_H = int(H * np.random.uniform(self.local_crop_ratio, 1.0))
                crop_W = int(W * np.random.uniform(self.local_crop_ratio, 1.0))

                # Random offset from center
                max_offset_D = int(D * self.max_offset_fraction)
                max_offset_H = int(H * self.max_offset_fraction)
                max_offset_W = int(W * self.max_offset_fraction)

                offset_D = np.random.randint(-max_offset_D, max_offset_D + 1)
                offset_H = np.random.randint(-max_offset_H, max_offset_H + 1)
                offset_W = np.random.randint(-max_offset_W, max_offset_W + 1)

                center_D, center_H, center_W = D // 2, H // 2, W // 2
                start_D = np.clip(center_D - crop_D // 2 + offset_D, 0, D - crop_D)
                start_H = np.clip(center_H - crop_H // 2 + offset_H, 0, H - crop_H)
                start_W = np.clip(center_W - crop_W // 2 + offset_W, 0, W - crop_W)

                cropped = tensor[:, start_D:start_D + crop_D,
                                    start_H:start_H + crop_H,
                                    start_W:start_W + crop_W]
                
                # Resize back to original shape
                resize_transform = tio.Resize((D, H, W))
                cropped_image = tio.Image(tensor=cropped, affine=affine)
                resized_image = resize_transform(cropped_image)
                # Wrap back into a Subject
                return tio.Subject(image=resized_image)

        # Pass the arguments here
        return CenterCropAndResizeTransform(
            local_crop_ratio=self.local_crop_ratio,
            max_offset_fraction=self.max_offset_fraction
        )

        # class CenterCropAndResizeTransform(tio.Transform):
        #     def __init__(self, crop_retention):
        #         super().__init__()
        #         self.crop_retention = crop_retention

        #     def apply_transform(self, buffer):
        #         # buffer shape: (C, H, W, T)
        #         C, H, W, T = buffer.shape
                
        #         # Compute crop size for H and W dimensions
        #         retention_factor = random.uniform(self.crop_retention, 1.0)
        #         crop_H = int(H * retention_factor)
        #         crop_W = int(W * retention_factor)

        #         # Crop or pad the H and W dimensions only
        #         crop = tio.CropOrPad(target_shape=(crop_H, crop_W, T))
        #         cropped = crop(buffer.permute(1, 2, 3, 0))  # Permute to (H, W, T, C)

        #         # Resize the cropped H and W dimensions back to the original size
        #         resize = tio.Resample(target_shape=(H, W, T))
        #         resized = resize(cropped)

        #         # Permute back to the original shape (C, H, W, T)
        #         resized_buffer = resized.permute(3, 0, 1, 2)
                    
        #         return resized_buffer

        # return CenterCropAndResizeTransform(self.crop_retention)


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _tensor_normalize_inplace(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize (with dimensions C, T, H, W).
        mean (tensor): mean value to subtract (in 0 to 255 floats).
        std (tensor): std to divide (in 0 to 255 floats).
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()

    C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)  # Make C the last dimension
    tensor.sub_(mean).div_(std)
    tensor = tensor.permute(1, 0).view(C, T, H, W)  # Put C back in front
    return tensor
