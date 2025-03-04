# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torchvision.transforms as transforms

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

def make_transforms(
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(1.0,1.0), #(3/4, 4/3),
    random_resize_scale=(0.9, 1.0),
    rot_degree = 0.0,
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size= 224,
    intensity_gamma=0.2,
    random_bias=0.2,
    random_noise=0.025,
    #normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    normalize=((0.0),(1))
):

   # _frames_augmentation = VideoTransform(
    _frames_augmentation = MRITransform(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        auto_augment=auto_augment,
        crop_size=crop_size,
        random_resize_scale=random_resize_scale,
        rot_degree = rot_degree,
        intensity_gamma=intensity_gamma,
        random_bias=random_bias,
        random_noise=random_noise
    )
    return _frames_augmentation

class MRITransform(object):

    def __init__(
        self,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(1.0,1.0), #(0.9, 1.1),
        random_resize_scale=(0.8,1.0), # use for crop_retention ratio
        rot_degree = 0.0, 
        auto_augment=False,
        crop_size=224,
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
        self.crop_retention=random_resize_scale[0]
        self.intensity_gamma=intensity_gamma
        self.random_bias=random_bias
        self.random_noise=random_noise
  
    def __call__(self, buffer):

        buffer = torch.tensor(buffer, dtype=torch.float32)
        # buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W
        
        if self.auto_augment:
            # buffer = np.transpose(buffer, (3, 1, 2, 0))  # T H W C -> C H W T
            # Permute to shape C H W T for TorchIO compatibility
            buffer = buffer.permute(3, 1, 2, 0)  # T H W C -> C H W T
            
            # Define the MRI spatial transformation list
            spatial_transforms = {
                tio.RandomAffine(
                    scales=self.random_resize_aspect_ratio,
                    degrees=self.rot_degree,
                ),  # Random affine transformation

                tio.RandomFlip(axes=('LR',)),  # Flip along the left-right axis

                # tio.RandomElasticDeformation(num_control_points=9),  # Elastic deformation
            }

            # Define the MRI intensity transformation list
            intensity_transforms = {
                tio.RandomGamma(log_gamma=(-self.intensity_gamma,self.intensity_gamma)),  # Random gamma adjustment

                # tio.RandomBiasField(coefficients=self.random_bias),  # Random bias field artifact

                tio.RandomNoise(mean=0.0, std=self.random_noise),  # Add random Gaussian noise
            }
            # Combine MRI spatial transforms using OneOf
            # transform = tio.OneOf(transforms_dict)
            # buffer = transform(buffer)

            # Combine spatial and intensity transforms using OneOf
            spatial_transform = tio.OneOf(spatial_transforms)
            intensity_transform = tio.OneOf(intensity_transforms)

            # Apply the transforms
            buffer = spatial_transform(buffer)
            buffer = intensity_transform(buffer)
    
            # Permute back to original shape T H W C
            buffer = buffer.permute(3, 1, 2, 0)  # C H W T ->  T H W C


        # if self.auto_augment:
        #     buffer = buffer.permute(3, 1, 2, 0)  # T H W C -> C H W T
        #     # Apply transformations to the buffer
        #     buffer = transforms(buffer)

        #     buffer = buffer.permute(3, 1, 2, 0)  # C H W T -> T H W C
            
        #GU_ debug
        # mid_slice_index = buffer.shape[0] // 2  # Compute the middle slice index along the temporal axis
        # plt.imsave('xformedBuffer.png', buffer[mid_slice_index, :, :,0].cpu().numpy(), cmap='gray')
        #GU_ debug
        # affine = np.eye(4)
        # nifti_image = nib.Nifti1Image(buffer.numpy(), affine)
        # nib.save(nifti_image, 'xformed_volume.nii')

        return buffer

    def custom_center_crop(self):
        """
        Creates a custom center crop transform that retains 90-100% of the input size
        on the H and W dimensions, and resizes back to the original H and W size.
        """
        class CenterCropAndResizeTransform(tio.Transform):
            def __init__(self, crop_retention):
                super().__init__()
                self.crop_retention = crop_retention

            def apply_transform(self, buffer):
                # buffer shape: (C, H, W, T)
                C, H, W, T = buffer.shape

                # Compute crop size for H and W dimensions
                retention_factor = random.uniform(self.crop_retention, 1.0)
                crop_H = int(H * retention_factor)
                crop_W = int(W * retention_factor)

                # Crop or pad the H and W dimensions only
                crop = tio.CropOrPad(target_shape=(crop_H, crop_W, T))
                cropped = crop(buffer.permute(1, 2, 3, 0))  # Permute to (H, W, T, C)

                # Resize the cropped H and W dimensions back to the original size
                resize = tio.Resample(target_shape=(H, W, T))
                resized = resize(cropped)

                # Permute back to the original shape (C, H, W, T)
                resized_buffer = resized.permute(3, 0, 1, 2)

                return resized_buffer

        return CenterCropAndResizeTransform(self.crop_retention)

class VideoTransform(object):

    def __init__(
        self,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3/4, 4/3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
    ):

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_size = crop_size
        self.mean = torch.tensor(normalize[0], dtype=torch.float32)
        self.std = torch.tensor(normalize[1], dtype=torch.float32)
        if not self.auto_augment:
            # Without auto-augment, PIL and tensor conversions simply scale uint8 space by 255.
            self.mean *= 255.
            self.std *= 255.

        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

        self.spatial_transform = video_transforms.random_resized_crop_with_shift \
            if motion_shift else video_transforms.random_resized_crop

        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode='pixel',
            max_count=1,
            num_splits=1,
            device='cpu',
        )

    def __call__(self, buffer):

        if self.auto_augment:
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
            buffer = self.autoaug_transform(buffer)
            buffer = [transforms.ToTensor()(img) for img in buffer]
            buffer = torch.stack(buffer)  # T C H W
            buffer = buffer.permute(0, 2, 3, 1)  # T H W C
        else:
            buffer = torch.tensor(buffer, dtype=torch.float32)

        buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W

        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )
        if self.random_horizontal_flip:
            buffer, _ = video_transforms.horizontal_flip(0.5, buffer)

        buffer = _tensor_normalize_inplace(buffer, self.mean, self.std)
        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


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
