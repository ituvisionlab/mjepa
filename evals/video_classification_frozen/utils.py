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
import io

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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb
from io import BytesIO
from PIL import Image

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
        _, N, D = outputs.size() #num_clips, B

        T = T // self.tubelet_size  # Num temporal tokens
        N = N // T  # Num spatial tokens

        # Unroll outputs into a 2D array [spatial_views x temporal_views]: num_views_per_clip, num_clips, B, temporal patch#, spatical patch#, dim d
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
            outputs = [o.reshape(B, T, N, D) for o in outputs]
            outputs = torch.cat(outputs, dim=1).flatten(1, 2)

            # Compute positional embedding
            if (self.pos_embed is not None) and (clip_indices is not None):
                clip_indices = [c[:, ::self.tubelet_size] for c in clip_indices]
                pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, F, D]
                pos_embed = apply_masks(pos_embed, clip_indices, concat=False)  # list(Tensor([B, T, D]))
                pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, N, 1)  # [B, T*num_clips, N, D]
                pos_embed = pos_embed.flatten(1, 2)
                outputs += pos_embed

            all_outputs[i] = outputs

        return all_outputs

class ChannelAggregation(nn.Module):
    """
    Process each channel independently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        tubelet_size=8,
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
        _, N, D = outputs.size() #num_clips, B

        T = T // self.tubelet_size  # Num temporal tokens
        N = N // T  # Num spatial tokens

        # Unroll outputs into a 2D array [spatial_views x temporal_views]: num_views_per_clip, num_clips, B, temporal patch#, spatical patch#, dim d
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
            outputs = [o.reshape(B, T, N, D) for o in outputs]
            outputs = torch.cat(outputs, dim=1).flatten(1, 2)

            # Compute positional embedding
            if (self.pos_embed is not None) and (clip_indices is not None):
                clip_indices = [c[:, ::self.tubelet_size] for c in clip_indices]
                pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, F, D]
                pos_embed = apply_masks(pos_embed, clip_indices, concat=False)  # list(Tensor([B, T, D]))
                pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, N, 1)  # [B, T*num_clips, N, D]
                pos_embed = pos_embed.flatten(1, 2)
                outputs += pos_embed

            all_outputs[i] = outputs

        return all_outputs
    
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
    random_blur=(0.01, 0.02),
    num_views_per_clip=1,
    in_chans=1,
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
            random_noise=random_noise,
            random_blur=random_blur,
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
        random_blur=(0.01, 0.02),
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
        self.random_blur=random_blur

    def __call__(self, buffer):

        buffer = torch.tensor(buffer, dtype=torch.float32) #T H W C      
        
        if self.auto_augment:
            buffer = buffer.permute(3, 1, 2, 0)  # T H W C -> C H W T
            
            subject_dict = {
                f"modality_{i}": tio.Image(tensor=buffer[i:i+1], type=tio.INTENSITY)
                for i in range(buffer.shape[0])
            }
            subject = tio.Subject(subject_dict)

            # Combine center crop and affine transforms into one spatial pool
            spatial_transform = tio.OneOf({
                # self.custom_center_crop(): 0.25,  # gently crop 25% of the time
                tio.RandomAffine(scales=self.random_resize_aspect_ratio, degrees=self.rot_degree): 0.3,
                tio.RandomFlip(axes=('LR',)): 0.3,
                tio.Lambda(lambda x: x): 0.4,
            })

            intensity_transform = tio.OneOf({
                tio.RandomGamma(log_gamma=(-self.intensity_gamma, self.intensity_gamma)): 0.2,
                tio.RandomNoise(mean=0.0, std=self.random_noise): 0.2,
                tio.RandomBlur(std=self.random_blur): 0.2,
                tio.Lambda(lambda x: x): 0.4,
            })

            # Apply full transform
            full_transform = tio.Compose([
                spatial_transform,
                intensity_transform,
            ])

            subject = full_transform(subject)

            # Reassemble tensor from subject
            buffer = torch.cat([subject[f"modality_{i}"].tensor for i in range(buffer.shape[0])], dim=0)
            buffer = buffer.permute(3, 1, 2, 0)  # [C, H, W, T] → [T, H, W, C]

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

@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_batches=10):
    """
    Extract embeddings from a model over a dataloader.
    """
    model.eval()
    embeddings = []
    labels = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        x = batch[0].to(device)
        feats = model(x)
        embeddings.append(feats.cpu())
        if len(batch) > 1:
            labels.append(batch[1])
    all_embeddings = torch.cat(embeddings)
    all_labels = torch.cat(labels) if labels else None
    return all_embeddings, all_labels


def plot_tsne(embeddings, labels=None, save_path=None, method='tsne', title='TSNE Embedding Visualization', wandb_log=False, wandb_key="tsne/embeddings", max_plot_samples=500):
    """
    Perform dimensionality reduction (t-SNE or PCA) and visualize embeddings.
    
    Args:
        embeddings (Tensor): High-dimensional embeddings [N, D]
        labels (Tensor or None): Optional labels for coloring
        method (str): 'tsne' or 'pca'
        title (str): Plot title
        save_path (str or None): If provided, saves plot to this path
        wandb_log (bool): If True, logs image to wandb
        wandb_key (str): Key name for wandb
        step (int or None): Training step or epoch
    """
    print(f"Original number of embeddings for t-SNE: {embeddings.shape[0]}")

    # Subsample if too many points
    if embeddings.shape[0] > max_plot_samples:
        print(f"Subsampling {max_plot_samples} points randomly for TSNE visualization.")
        indices = np.random.choice(embeddings.shape[0], size=max_plot_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    reducer = TSNE(n_components=2, perplexity=30, init='pca', random_state=42) if method == 'tsne' else PCA(n_components=2) #init='random'
    reduced = reducer.fit_transform(embeddings)

    # Create a scatter plot
    unique_labels = np.unique(labels)
    markers = ['o', '+', 'x', 's', 'd', '^', 'v', '<', '>', 'p', '*']
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8)
    else:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], s=5)
    plt.colorbar(scatter)

    # for idx, label in enumerate(unique_labels):
    #     indices = labels == label
    #     plt.scatter(reduced[indices, 0], reduced[indices, 1],
    #                 marker=markers[idx % len(markers)],
    #                 color=colors(idx % 10),
    #                 label=f'Class {label}', alpha=0.7)
    #plt.legend()
   
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.tight_layout()

    # Save locally if needed
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to: {save_path}")

    # Save to wandb if requested
    if wandb_log and wandb.run is not None:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)  # Convert buffer to PIL image
        wandb.log({wandb_key: wandb.Image(image, caption=title)})
        buf.close()
        #wandb.log({wandb_key: wandb.Image(save_path)})
        print(f"Logged plot to wandb: {wandb_key}")

    plt.close()
