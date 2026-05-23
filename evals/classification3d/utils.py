# mjepa: A 3D MRI self-supervised learning framework based on a modified V-JEPA
# Copyright (c) 2024–2025 [Gozde Unal, NYU]
#
# This file is based on an earlier version of code from:
# V-JEPA (https://github.com/facebookresearch/v-jepa)
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This codebase has been significantly modified for use in medical imaging and 3D MRI.
# All modifications are licensed under the original MIT license (or the applicable license).
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchio as tio
import random
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms

from src.datasets.utils.video.randerase import RandomErasing

from src.models.utils.pos_embs import get_1d_sincos_pos_embed
from src.masks.utils import apply_masks

import matplotlib.pyplot as plt

class FrameAggregation(nn.Module):
    """
    Process each frame independently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        max_frames=10000,
        use_pos_embed=False,
        attend_across_segments=False
    ):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.attend_across_segments = attend_across_segments
        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_frames, embed_dim),
                requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_frames)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):

        # TODO: implement attend_across_segments=False
        # num_clips = len(x)
        num_views_per_clip = len(x[0])

        # Concatenate views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        # Concatenate clips along temporal dimension
        x = torch.cat(x, dim=2)
        B, C, T, H, W = x.size()

        # Put each frame along the batch dimension
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

        outputs = self.model(x)
        _, N, D = outputs.size()
        outputs = outputs.reshape(B, T, N, D).flatten(1, 2)

        # Separate views into list
        B = B // num_views_per_clip
        all_outputs = []
        for i in range(num_views_per_clip):
            o = outputs[i*B:(i+1)*B]
            # Compute positional embedding
            if (self.pos_embed is not None) and (clip_indices is not None):
                pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, F, D]
                pos_embed = apply_masks(pos_embed, clip_indices, concat=False)  # list(Tensor([B, T, D]))
                pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, N, 1)  # [B, T*num_clips, N, D]
                pos_embed = pos_embed.flatten(1, 2)
                o += pos_embed
            all_outputs += [o]

        return all_outputs


class ClipAggregation(nn.Module):
    """
    Process each clip independently and concatenate all tokens
    """

    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model
 
    def forward(self, x):
        # If x is already a tensor, just forward it
        if isinstance(x, torch.Tensor):
            return self.model(x)
        # else: assume list-of-list structure from multiview pipeline
        #  Concatenate all spatial and temporal views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)
        return self.model(x)


class LayerAggregation(nn.Module):
    """
    Takes last n layers of encoder features and average pools them along patch dimension
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=10000,
        use_pos_embed=False,
        attend_across_segments=False
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.attend_across_segments = attend_across_segments
        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_T, embed_dim),
                requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):
        
        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, T, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)
        outputs = self.model(x)
        
        layer_outputs = outputs
        all_layer_outputs = []
        
        for outputs in layer_outputs:
        
            _, N, D = outputs.size()

            new_T = T // self.tubelet_size  # Num temporal tokens
            N = N // new_T  # Num spatial tokens

            # Unroll outputs into a 2D array [spatial_views x temporal_views]
            eff_B = B * num_views_per_clip
            all_outputs = [[] for _ in range(num_views_per_clip)]
            for i in range(num_clips):
                o = outputs[i*eff_B:(i+1)*eff_B]
                for j in range(num_views_per_clip):
                    all_outputs[j].append(o[j*B:(j+1)*B])

            if not self.attend_across_segments:
                return all_outputs

            for i, outputs in enumerate(all_outputs):

                # Concatenate along temporal dimension
                outputs = [o.reshape(B, new_T, N, D) for o in outputs]
                outputs = torch.cat(outputs, dim=1).flatten(1, 2)
                
                all_layer_outputs.append(outputs)
        
        
        pooled_outs = []
        for layer in all_layer_outputs:
            out = torch.mean(layer, dim=1)
            out = out.squeeze(1)
            pooled_outs.append(out)
        
        outputs = torch.cat(pooled_outs, dim=-1)
        
        return outputs
        
        # num_clips = len(x)
        # num_views_per_clip = len(x[0])
        # B, C, T, H, W = x[0][0].size()
        
        # x = [torch.cat(xi, dim=0) for xi in x]
        # x = torch.cat(x, dim=0)
        # outputs = self.model(x)
        
        # eff_B = B
        # all_outputs = []
        # for i in range(num_clips):
        #     o = outputs[i*eff_B:(i+1)*eff_B]
        #     for j in range(num_views_per_clip):
        #         all_outputs[j].append(o[j*B:(j+1)*B])
                
        # layer_outs = []
        
        # for layer in outputs:
        #     out = torch.mean(layer, dim=1)
        #     out = out.squeeze(1)
        #     layer_outs.append(out)
        
        # outputs = torch.cat(layer_outs, dim=-1)
        
        # return outputs

def make_video_transforms(
    training=True,
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3/4, 4/3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    num_views_per_clip=1,
    normalize=((0.485, 0.456, 0.406),
               (0.229, 0.224, 0.225))
):

    if not training and num_views_per_clip > 1:
        print('Making EvalVideoTransform, multi-view')
        _frames_augmentation = EvalVideoTransform(
            num_views_per_clip=num_views_per_clip,
            short_side_size=crop_size,
            normalize=normalize,
        )

    else:
        _frames_augmentation = VideoTransform(
            training=training,
            random_horizontal_flip=random_horizontal_flip,
            random_resize_aspect_ratio=random_resize_aspect_ratio,
            random_resize_scale=random_resize_scale,
            reprob=reprob,
            auto_augment=auto_augment,
            motion_shift=motion_shift,
            crop_size=crop_size,
            normalize=normalize,
        )
    return _frames_augmentation


def make_transforms(
    training=True,
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(1.0,1.0), #(3/4, 4/3),
    random_resize_scale=(0.9, 1.0),
    rot_degree = 0.0,
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    intensity_gamma=0.2,
    random_bias=0.2,
    random_noise=0.025,
    num_views_per_clip=1,
    in_chans=3,
    #normalize=((0.485, 0.456, 0.406),
    #           (0.229, 0.224, 0.225))
    normalize=((0.0),(1))
):

    if not training: # and num_views_per_clip > 1:  # GU_
        print('Making EvalMRITransform, multi-view')
        _frames_augmentation = EvalMRITransform(
            num_views_per_clip=num_views_per_clip,
            short_side_size=crop_size
        )

    else:
        _frames_augmentation = MRITransform(
            training=training,
            random_horizontal_flip=random_horizontal_flip,
            random_resize_aspect_ratio=random_resize_aspect_ratio,
            random_resize_scale=random_resize_scale,
            rot_degree = rot_degree,
            auto_augment=auto_augment,
            crop_size=crop_size,
            intensity_gamma=intensity_gamma,
            random_bias=random_bias,
            random_noise=random_noise
        )
    return _frames_augmentation

class MRITransform(object):

    def __init__(
        self,
        training=True,
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

        self.training = training

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

        buffer = torch.tensor(buffer, dtype=torch.float32) #T H W C      
        
        if self.auto_augment:
            buffer = buffer.permute(3, 1, 2, 0)  # T H W C -> C H W T
            
            # Define the MRI spatial transformation list
            spatial_transforms = {
                tio.RandomAffine(
                    scales=self.random_resize_aspect_ratio,
                    degrees=self.rot_degree,
                ),  # Random affine transformation

                tio.RandomFlip(axes=('LR',)),  # Flip along the left-right axis

                tio.RandomElasticDeformation(num_control_points=9),  # Elastic deformation
            }

            # Define the MRI intensity transformation list
            intensity_transforms = {
                tio.RandomGamma(log_gamma=(-self.intensity_gamma,self.intensity_gamma)),  # Random gamma adjustment

                tio.RandomBiasField(coefficients=self.random_bias),  # Random bias field artifact

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

            buffer = buffer.permute(3, 1, 2, 0)  # C H W T ->  T H W C

        # buffer = buffer.permute(3, 0, 1, 2)  # T H W C --> C T H W

        return [buffer]

class EvalMRITransform(object):

    def __init__(
        self,
        num_views_per_clip=1,
        short_side_size=224,
    ):
        self.views_per_clip = num_views_per_clip
        self.short_side_size = short_side_size
        # self.spatial_resize = video_transforms.Resize(short_side_size, interpolation='bilinear')
        # self.to_tensor = video_transforms.Compose([
        #     volume_transforms.ClipToTensor(channel_nb=in_chans),
        #     # video_transforms.Normalize(mean=normalize[0], std=normalize[1]) #GU_COMMENT
        # ])

    def __call__(self, buffer):

        buffer = np.array(buffer) #T W H C
        T, H, W, C = buffer.shape

        buffer = torch.tensor(buffer, dtype=torch.float32) #T H W C
       #  buffer = buffer.permute(3, 0, 1, 2)  # T H W C --> C T H W

        return [buffer]
        
        # num_views = self.views_per_clip
        # side_len = self.short_side_size
        # spatial_step = (max(H, W) - side_len) // (num_views - 1) # GU_

        # all_views = []
        
        #GU_COMMENT
        #for i in range(num_views):
        #    start = i*spatial_step
        #    if H > W:
        #        view = buffer[:, start:start+side_len, :, :]
        #    else:
        #        view = buffer[:, :, start:start+side_len, :]
        #    view = self.to_tensor(view)
        #    all_views.append(view)
        # view = self.to_tensor(buffer)    
        # all_views.append(view)
        
        # all_views.append(buffer)
        # return all_views

class VideoTransform(object):

    def __init__(
        self,
        training=True,
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

        self.training = training

        short_side_size = int(crop_size * 256 / 224)
        self.eval_transform = video_transforms.Compose([
            video_transforms.Resize(short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop(size=(crop_size, crop_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_size = crop_size
        self.normalize = torch.tensor(normalize)

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

        if not self.training:
            return [self.eval_transform(buffer)]

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        if self.auto_augment:
            buffer = self.autoaug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        buffer = tensor_normalize(buffer, self.normalize[0], self.normalize[1])
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

        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return [buffer]


class EvalVideoTransform(object):

    def __init__(
        self,
        num_views_per_clip=1,
        short_side_size=224,
        normalize=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
    ):
        self.views_per_clip = num_views_per_clip
        self.short_side_size = short_side_size
        self.spatial_resize = video_transforms.Resize(short_side_size, interpolation='bilinear')
        self.to_tensor = video_transforms.Compose([
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])

    def __call__(self, buffer):

        # Sample several spatial views of each clip
        buffer = np.array(self.spatial_resize(buffer))
        T, H, W, C = buffer.shape

        num_views = self.views_per_clip
        side_len = self.short_side_size
        spatial_step = (max(H, W) - side_len) // (num_views - 1)

        all_views = []
        for i in range(num_views):
            start = i*spatial_step
            if H > W:
                view = buffer[:, start:start+side_len, :, :]
            else:
                view = buffer[:, :, start:start+side_len, :]
            view = self.to_tensor(view)
            all_views.append(view)

        return all_views


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
