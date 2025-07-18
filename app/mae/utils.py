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
import numpy as np
import nibabel as nib
from datetime import datetime
from scipy.stats import entropy

import src.models.vision_transformer as video_vit
import src.models.decoder as vit_decoder
from src.models.utils.multimask import MultiMaskWrapper #DecoderMultiMaskWrapper
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
    pred_num_heads=None,
    in_chans=3,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=1,
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
        num_heads=encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
        in_chans=in_chans,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate
    )
    # decoder = DecoderMultiMaskWrapper(decoder) #Remove multimask wrapper for MAE

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
    decoder_lr_scale=1.0,
):
    param_groups = [
    {
        'params': (p for n, p in encoder.named_parameters()
                   if ('bias' not in n) and (len(p.shape) != 1)),
        'name': 'encoder_weight'
    }, {
        'params': (p for n, p in decoder.named_parameters()
                   if ('bias' not in n) and (len(p.shape) != 1)),
        'lr': ref_lr * decoder_lr_scale,  # ← scaled LR for decoder
        'name': 'decoder_weight'
    }, {
        'params': (p for n, p in encoder.named_parameters()
                   if ('bias' in n) or (len(p.shape) == 1)),
        'WD_exclude': zero_init_bias_wd,
        'weight_decay': 0,
        'name': 'encoder_bias'
    }, {
        'params': (p for n, p in decoder.named_parameters()
                   if ('bias' in n) or (len(p.shape) == 1)),
        'WD_exclude': zero_init_bias_wd,
        'weight_decay': 0,
        'lr': ref_lr * decoder_lr_scale,  # ← scaled LR for decoder bias/etc
        'name': 'decoder_bias'
    },
    ]

    # param_groups = [
    #     {
    #         'params': (p for n, p in encoder.named_parameters()
    #                    if ('bias' not in n) and (len(p.shape) != 1))
    #     }, {
    #         'params': (p for n, p in decoder.named_parameters()
    #                    if ('bias' not in n) and (len(p.shape) != 1))
    #     }, {
    #         'params': (p for n, p in encoder.named_parameters()
    #                    if ('bias' in n) or (len(p.shape) == 1)),
    #         'WD_exclude': zero_init_bias_wd,
    #         'weight_decay': 0,
    #     }, {
    #         'params': (p for n, p in decoder.named_parameters()
    #                    if ('bias' in n) or (len(p.shape) == 1)),
    #         'WD_exclude': zero_init_bias_wd,
    #         'weight_decay': 0,
    #     },
    # ]

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

    # Reshape the full tokens (cuurently B, N, D tensor) into a video volume.
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

    # Since mRI volumes are 1-channel and we only want the first sample in the batch, for visualization
    # return the first sample and remove the channel dimension.
    # This yields a 3D tensor with shape (num_frames, crop_size, crop_size).
    # return full_tokens[0] # shape: [1, T, H, W]: from 1st sample for visualization
    return full_tokens #[C, T, H, W]:

def unpatchify_image_from_full(full_tokens, patch_size, tubelet_size, num_frames, in_chans, crop_size):
    """
    Reconstruct a full video volume from patch tokens.

    Returns:
        Tensor of shape [B, C, T, H, W]
    """
    B, L, D = full_tokens.shape
    grid_spatial = crop_size // patch_size
    grid_temporal = num_frames // tubelet_size
    assert grid_spatial * grid_spatial * grid_temporal == L, \
        f"Expected {grid_spatial ** 2 * grid_temporal} patches, got {L}"

    full_tokens = full_tokens.view( 
        B, grid_temporal, grid_spatial, grid_spatial,
        tubelet_size, patch_size, patch_size, in_chans
    )

    full_tokens = full_tokens.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous() # shape [B, C, T, H, W]
    return full_tokens.view(
        B,
        in_chans,
        grid_temporal * tubelet_size,
        grid_spatial * patch_size,
        grid_spatial * patch_size
    )

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

# saving of fully-reconstructed volumes
def save_volume_with_log(volume, affine, save_path, log_path):
    try:
        with open(log_path, "a") as log_file:
            log_file.write(f"\n--- Saving Volume: {save_path} ---\n")
            log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")

            # Check for NaNs or Infs
            has_nan = np.isnan(volume).any()
            has_inf = np.isinf(volume).any()
            if has_nan or has_inf:
                log_file.write(f"[WARNING] Volume has NaNs: {has_nan}, Infs: {has_inf}. Replaced with 0.\n")
                volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

            # Check dtype
            if volume.dtype not in [np.float32, np.float64, np.int16, np.uint8]:
                log_file.write(f"[WARNING] Unsupported dtype {volume.dtype}. Converting to float32.\n")
                volume = volume.astype(np.float32)

            # Check shape
            if volume.ndim != 3:
                log_file.write(f"[ERROR] Invalid shape: {volume.shape}. Must be 3D.\n")
                return False

            # Save volume
            img = nib.Nifti1Image(volume, affine)
            nib.save(img, save_path)

            # Reload immediately and validate
            try:
                img_reloaded = nib.load(save_path)
                consistent_shape = img_reloaded.shape == volume.shape
                consistent_affine = np.allclose(img_reloaded.affine, affine)

                if not consistent_shape:
                    log_file.write(f"[ERROR] Shape mismatch after reload: Original {volume.shape}, Reloaded {img_reloaded.shape}\n")
                    return False
                if not consistent_affine:
                    log_file.write(f"[ERROR] Affine mismatch after reload.\n")
                    return False

                log_file.write("[SUCCESS] Volume saved, reloaded, and verified successfully.\n")
                return True

            except Exception as reload_exc:
                log_file.write(f"[ERROR] Reload failed: {reload_exc}\n")
                return False

    except Exception as exc:
        with open(log_path, "a") as log_file:
            log_file.write(f"[CRITICAL ERROR] Unexpected failure: {exc}\n")
        return False

def sanity_check_recons_volumes(volume, affine, save_path, log_path):

    vol_full_recon = nib.load('ZReconstructed_full_volume.nii.gz').get_fdata()
    vol_mosaic_recon = nib.load('ZReconstructed_mosaic_volume.nii.gz').get_fdata()

    hist_full, _ = np.histogram(vol_full_recon, bins=256, density=True)
    hist_mosaic, _ = np.histogram(vol_mosaic_recon, bins=256, density=True)

    entropy_full = entropy(hist_full)
    entropy_mosaic = entropy(hist_mosaic)

    print(f"Full Recon Entropy: {entropy_full}")
    print(f"Mosaic Recon Entropy: {entropy_mosaic}")

    nib.save(vol_full_recon, 'full_uncompressed.nii')
    nib.save(vol_mosaic_recon, 'mosaic_uncompressed.nii')