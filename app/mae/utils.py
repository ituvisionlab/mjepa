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

import torch
import matplotlib.pyplot as plt

import src.models.vision_transformer as video_vit
import src.models.decoder as vit_decoder
from src.models.utils.multimask import MultiMaskWrapper, DecoderMultiMaskWrapper
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    r_path,
    encoder,
    decoder,
    opt,
    scaler,
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
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['decoder']
        msg = decoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return (
        encoder,
        decoder,
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
    pred_model_name='vit_decoder',
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
    decoder = vit_decoder.__dict__[pred_model_name](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
        in_chans=in_chans,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate
    )
    decoder = DecoderMultiMaskWrapper(decoder)

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
    decoder.to(device)
    logger.info(encoder)
    logger.info(decoder)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'Encoder number of parameters: {count_parameters(encoder)}')
    logger.info(f'Decoder number of parameters: {count_parameters(decoder)}')

    return encoder, decoder


def init_opt(
    encoder,
    decoder,
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
            'params': (p for n, p in decoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': zero_init_bias_wd,
            'weight_decay': 0,
        }, {
            'params': (p for n, p in decoder.named_parameters()
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


def patchify_image(x, patch_size):
    """
    ATTENTION!!!!!!!
    Different from 2D version patchification: The final axis follows the order of [ph, pw, pd, c] instead of [c, ph, pw, pd]
    """
    # patchify input, [B,C,D,H,W] --> [B,C,gd,pd,gh,ph,gw,pw] --> [B,gd*gh*gw,pd*ph*pw*C]
    B, C, D, H, W = x.shape
    
    # Handle patch size as tuple or single value
    if isinstance(patch_size, (tuple, list)):
        ph, pw, pd = patch_size
    else:
        ph = pw = pd = patch_size
        
    grid_size = (D // pd, H // ph, W // pw)

    x = x.reshape(B, C, grid_size[0], pd, grid_size[1], ph, grid_size[2], pw) # [B,C,gd,pd,gh,ph,gw,pw]
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, grid_size[0] * grid_size[1] * grid_size[2], pd * ph * pw * C) # [B,gd*gh*gw,pd*ph*pw*C]

    return x

def unpatchify_image(recon, nonmask, patch_size, tubelet_size, num_frames, in_chans, crop_size, mask_enc, mask_pred):
    """
    Reconstruct a full video volume from patch tokens for a given mask level.

    Args:
        recon (torch.Tensor): Reconstructed (masked) tokens,
            shape (B, L_masked, patch_size**2 * tubelet_size * in_chans).
        nonmask (torch.Tensor): Original (unmasked) tokens,
            shape (B, L_unmasked, patch_size**2 * tubelet_size * in_chans).
        patch_size (int): Spatial patch size.
        tubelet_size (int): Temporal patch (tubelet) size.
        num_frames (int): Total number of frames in the video.
        in_chans (int): Number of channels.
        crop_size (int): Spatial crop size of the video (assumed square).
        mask_enc (torch.Tensor): Tensor of indices for unmasked tokens for this level (B, L_unmasked).
        mask_pred (torch.Tensor): Tensor of indices for masked tokens for this level (B, L_masked).

    Returns:
        torch.Tensor: Reconstructed video volume of shape
            (B, in_chans, num_frames, crop_size, crop_size).
    """
    B = recon.shape[0]
    L_masked = recon.shape[1]
    L_unmasked = nonmask.shape[1]
    L = L_masked + L_unmasked  # Total number of patches

    # Compute grid sizes
    grid_spatial = crop_size // patch_size  # number of patches per spatial dimension
    grid_temporal = num_frames // tubelet_size
    assert grid_spatial * grid_spatial * grid_temporal == L, (
        f"Mismatch: Expected {grid_spatial * grid_spatial * grid_temporal} patches, got {L}"
    )

    # The flattened patch dimension
    D = patch_size * patch_size * tubelet_size * in_chans
    device = recon.device

    # Create a full token container with the same dtype as recon.
    full_tokens = torch.zeros(B, L, D, device=device, dtype=recon.dtype)

    # Ensure mask indices are of type long.
    mask_enc = mask_enc.long()
    mask_pred = mask_pred.long()

    # Make sure the token types match
    nonmask = nonmask.to(recon.dtype)

    # Scatter the unmasked tokens using mask_enc indices.
    full_tokens.scatter_(1, mask_enc.unsqueeze(-1).expand_as(nonmask), nonmask)
    # Scatter the reconstructed tokens using mask_pred indices.
    full_tokens.scatter_(1, mask_pred.unsqueeze(-1).expand_as(recon), recon)

    # Reshape the full tokens into a video volume.
    full_tokens = full_tokens.view(
        B,
        grid_temporal,    # temporal grid
        grid_spatial,     # spatial grid height
        grid_spatial,     # spatial grid width
        tubelet_size,     # temporal patch dimension
        patch_size,       # patch spatial height
        patch_size,       # patch spatial width
        in_chans          # channels
    )

    # Permute to (B, in_chans, num_frames, crop_size, crop_size)
    full_tokens = full_tokens.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    full_tokens = full_tokens.view(
        B,
        in_chans,
        grid_temporal * tubelet_size,   # num_frames
        grid_spatial * patch_size,        # height
        grid_spatial * patch_size         # width
    )

        # Since mRI volumes are 1-channel and we only want the first sample in the batch,
    # return the first sample and remove the channel dimension.
    # This yields a 3D tensor with shape (num_frames, crop_size, crop_size).
    return full_tokens[0, 0]

def unpatchify_image_from_full(full_tokens, patch_size, tubelet_size, num_frames, in_chans, crop_size):
    """
    Reconstruct a full video volume from patch tokens for a given mask level.

    Args:
        recon (torch.Tensor): Reconstructed (masked) tokens,
            shape (B, L_masked, patch_size**2 * tubelet_size * in_chans).
        patch_size (int): Spatial patch size.
        tubelet_size (int): Temporal patch (tubelet) size.
        num_frames (int): Total number of frames in the video.
        in_chans (int): Number of channels.
        crop_size (int): Spatial crop size of the video (assumed square).
    Returns:
        torch.Tensor: Reconstructed video volume of shape
            (B, in_chans, num_frames, crop_size, crop_size).
    """
    B = full_tokens.shape[0]
    L = full_tokens.shape[1]  # Total number of patches

    # Compute grid sizes
    grid_spatial = crop_size // patch_size  # number of patches per spatial dimension
    grid_temporal = num_frames // tubelet_size
    assert grid_spatial * grid_spatial * grid_temporal == L, (
        f"Mismatch: Expected {grid_spatial * grid_spatial * grid_temporal} patches, got {L}"
    )

    # Reshape the full tokens into a video volume.
    full_tokens = full_tokens.view(
        B,
        grid_temporal,    # temporal grid
        grid_spatial,     # spatial grid height
        grid_spatial,     # spatial grid width
        tubelet_size,     # temporal patch dimension
        patch_size,       # patch spatial height
        patch_size,       # patch spatial width
        in_chans          # channels
    )

    # Permute to (B, in_chans, num_frames, crop_size, crop_size)
    full_tokens = full_tokens.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    full_tokens = full_tokens.view(
        B,
        in_chans,
        grid_temporal * tubelet_size,   # num_frames
        grid_spatial * patch_size,        # height
        grid_spatial * patch_size         # width
    )

    # Since mRI volumes are 1-channel and we only want the first sample in the batch,
    # return the first sample and remove the channel dimension.
    # This yields a 3D tensor with shape (num_frames, crop_size, crop_size).
    return full_tokens[0, 0]

