# mjepa: A 3D MRI self-supervised learning framework based on a modified V-JEPA
# Copyright (c) 2024–2025 [Gozde Unal, NYU]
#
# This file is based on an earlier version of code from:
# V-JEPA (https://github.com/facebookresearch/v-jepa)
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This codebase has been significantly modified for use in medical imaging and 3D MRI.
# All modifications are licensed under the original MIT license (or the applicable license).

import logging
import sys
import warnings
import yaml
import os
import time
from collections import defaultdict

import torch
import matplotlib.pyplot as plt

import src.models.vision_transformer as video_vit
import src.models.predictor as vit_pred
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    discard_stem=False
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')

    epoch = 0
    try:
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        
        if discard_stem:
            # Remove patch_embed related weights
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "patch_embed" not in k and "pos_embed" not in k}
            logger.info("Ignoring patch_embed weights while loading encoder.")
            msg = encoder.load_state_dict(pretrained_dict, strict=False)
        else:
            msg = encoder.load_state_dict(pretrained_dict)
            
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        
        if discard_stem:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "pos_embed" not in k}
            msg = predictor.load_state_dict(pretrained_dict, strict=False)
        else:
            msg = predictor.load_state_dict(pretrained_dict)
        
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            
            if discard_stem:
                # Remove patch_embed related weights
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if "patch_embed" not in k and "pos_embed" not in k}
                logger.info("Ignoring patch_embed weights while loading encoder.")
                msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
            else:
                msg = target_encoder.load_state_dict(pretrained_dict)
            
            logger.info(
                f'loaded pretrained target encoder from epoch {epoch} with msg: {msg}'
            )

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        
        if discard_stem:
            opt.state = defaultdict(dict) #reset weight information of the optimizer
            
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
            
            if discard_stem:
                scaler.state = defaultdict(dict)
                
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    model_name='vit_base',
    pred_model_name='vit_predictor',
    crop_size=224,
    pred_depth=6,
    pred_embed_dim=384,
    in_chans=3,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    drop_rate=0.0,
    attn_drop_rate=0.0
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        in_chans=in_chans,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate
    )
    encoder = MultiMaskWrapper(encoder)
    predictor = vit_pred.__dict__[pred_model_name](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate
    )
    predictor = PredictorMultiMaskWrapper(predictor)

    # def init_weights(m):
    #     if isinstance(m, torch.nn.Linear):
    #         trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             torch.nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, torch.nn.LayerNorm):
    #         torch.nn.init.constant_(m.bias, 0)
    #         torch.nn.init.constant_(m.weight, 1.0)

    # for m in encoder.modules():
    #     init_weights(m)

    # for m in predictor.modules():
    #     init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'Encoder number of parameters: {count_parameters(encoder)}')
    logger.info(f'Predictor number of parameters: {count_parameters(predictor)}')

    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.0,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': zero_init_bias_wd,
            'weight_decay': 0,
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': zero_init_bias_wd,
            'weight_decay': 0,
        },
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler

def get_new_log_dir(root='./logs', postfix='', prefix=''):
    if root == None:
        return None
    log_dir = os.path.join(root, prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    os.makedirs(log_dir)
    return log_dir

import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_fft_3d_spectrum(feature, grid_shape, channel_idx=0, slice_dim=2, title_prefix=""):
    """
    Visualizes the log-magnitude FFT spectrum of a single feature channel from 3D embeddings.

    Args:
        feature: Tensor of shape [B, N, D] or [B, D, X, Y, Z]
        grid_shape: tuple of (X, Y, Z) if input is [B, N, D]
        channel_idx: index of feature channel D to visualize
        slice_dim: axis to slice through (0=X, 1=Y, 2=Z)
        title_prefix: prefix for figure title
    """
    if feature.ndim == 3:  # [B, N, D]
        B, N, D = feature.shape
        X, Y, Z = grid_shape
        assert N == X * Y * Z, f"Expected {X*Y*Z} tokens, got {N}"
        feature = feature.permute(0, 2, 1).contiguous().view(B, D, X, Y, Z)  # [B, D, X, Y, Z]

    B, D, X, Y, Z = feature.shape
    feat = feature[0, channel_idx]  # [X, Y, Z] of selected channel

    # Compute FFT and shift
    fft_vol = torch.fft.fftn(feat, dim=(0, 1, 2))
    fft_mag = torch.abs(torch.fft.fftshift(fft_vol))  # center DC component
    fft_log = torch.log1p(fft_mag).detach().cpu().numpy()

    # Take a central slice along chosen axis
    mid = fft_log.shape[slice_dim] // 2
    if slice_dim == 0:
        slice_img = fft_log[mid, :, :]
    elif slice_dim == 1:
        slice_img = fft_log[:, mid, :]
    else:
        slice_img = fft_log[:, :, mid]

    # Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(slice_img, cmap='inferno')
    plt.colorbar()
    plt.title(f"{title_prefix}FFT Spectrum Slice (channel {channel_idx}, axis {slice_dim})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
