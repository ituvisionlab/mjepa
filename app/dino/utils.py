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
import torch.nn.functional as F


import src.models.vision_transformer as video_vit
#import src.models.predictor as vit_pred
# from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    r_path,
    encoder,
    # predictor,
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
        msg = encoder.load_state_dict(pretrained_dict)
            
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        #pretrained_dict = checkpoint['predictor']
        #msg = predictor.load_state_dict(pretrained_dict)      
        #logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

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
        #predictor,
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
    crop_size=224,
    in_chans=3,
    uniform_power=False,
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
    # encoder = MultiMaskWrapper(encoder)

    # predictor = vit_pred.__dict__[pred_model_name](
    #     img_size=crop_size,
    #     use_mask_tokens=use_mask_tokens,
    #     patch_size=patch_size,
    #     num_frames=num_frames,
    #     tubelet_size=tubelet_size,
    #     embed_dim=encoder.backbone.embed_dim,
    #     predictor_embed_dim=pred_embed_dim,
    #     depth=pred_depth,
    #     num_heads=encoder.backbone.num_heads,
    #     uniform_power=uniform_power,
    #     num_mask_tokens=num_mask_tokens,
    #     zero_init_mask_tokens=zero_init_mask_tokens,
    #     use_sdpa=use_sdpa,
    #     drop_rate=drop_rate,
    #     attn_drop_rate=attn_drop_rate
    # )
    # predictor = PredictorMultiMaskWrapper(predictor)

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
    logger.info(encoder)
    #predictor.to(device)
    #logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'Encoder number of parameters: {count_parameters(encoder)}')
    # logger.info(f'Predictor number of parameters: {count_parameters(predictor)}')

    return encoder #, predictor


def init_opt(
    encoder,
    # predictor,
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
            'params': (p for n, p in encoder.named_parameters()
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


class DinoCenterManager:
    def __init__(self, feature_dim, device, momentum=0.9):
        self.center = torch.zeros(1, 1, feature_dim, device=device)
        self.momentum = momentum

    def update(self, teacher_output):
        # teacher_output: [B, N, D]
        batch_center = teacher_output.detach().mean(dim=(0, 1), keepdim=True)

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center)
            batch_center /= torch.distributed.get_world_size()

        self.center = self.center * self.momentum + batch_center * (1. - self.momentum)

    def get(self):
        return self.center

def cosine_similarity(student_output, teacher_output):
    """
    Compute average cosine similarity between student and teacher outputs.
    """
    student_norm = F.normalize(student_output, dim=-1)
    teacher_norm = F.normalize(teacher_output, dim=-1)
    return (student_norm * teacher_norm).sum(dim=-1).mean().item()

import torch
import torch.nn.functional as F

def dino_debug_dashboard(
    global_step,
    logger,
    z_g_student,
    h_teacher,
    center,
    prev_center,
    student_input_0,
    student_input_1,
    temperature_teacher,
    threshold_variance=0.005,
    threshold_entropy=5.0,
    threshold_kl=0.01,
    abort_on_collapse=False,
):
    # Cosine similarity
    cosine_sim = F.cosine_similarity(
        z_g_student[0].mean(dim=1), h_teacher[0].mean(dim=1), dim=-1
    ).mean().item()

    # Center tracking
    center_norm = center.norm().item()
    delta_center = (center - prev_center).norm().item()

    # Student STD per dimension
    z_flat = z_g_student[0].view(-1, z_g_student[0].shape[-1])
    std_per_dim = torch.sqrt(z_flat.var(dim=0) + 1e-4)
    std_mean = std_per_dim.mean().item()
    std_min = std_per_dim.min().item()
    std_max = std_per_dim.max().item()

    # Embedding variance per sample
    embedding_var = z_g_student[0].var(dim=1).mean().item()

    # View difference (in embedding space)
    view_diff = (student_input_0 - student_input_1).abs().mean().item()

    # Compute teacher softmax & diagnostics
    with torch.no_grad():
        teacher_logits = (h_teacher[0] - center) / temperature_teacher
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # Entropy
        entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-6), dim=-1).mean().item()

        # Max prob
        max_prob = teacher_probs.max(dim=-1)[0].mean().item()

        # Student log probs
        student_logits = z_g_student[0]
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # KL divergence
        kl_per_token = torch.sum(
            teacher_probs * (torch.log(teacher_probs + 1e-6) - student_log_probs),
            dim=-1
        )
        kl_div = kl_per_token.mean().item()

    # Logging
    logger.info(f"[DINO DEBUG @ step {global_step}]")
    logger.info(f"  Cosine Similarity (S vs T): {cosine_sim:.4f}")
    logger.info(f"  Teacher Entropy:            {entropy:.4f}")
    logger.info(f"  Teacher Max Prob:           {max_prob:.4f}")
    logger.info(f"  KL Divergence (S||T):       {kl_div:.4f}")
    logger.info(f"  Teacher Center Norm:        {center_norm:.4f}, Δcenter: {delta_center:.4f}")
    logger.info(f"  Student STD: min={std_min:.4f}, max={std_max:.4f}, mean={std_mean:.4f}")
    logger.info(f"  Student Embedding Variance: {embedding_var:.6f}")
    logger.info(f"  Global View Diff (emb):     {view_diff:.4f}")

    # Collapse detector
    if std_mean < threshold_variance or entropy > threshold_entropy or kl_div < threshold_kl:
        logger.warning(
            f"[COLLAPSE WARNING] Step {global_step}: variance={std_mean:.6f}, "
            f"entropy={entropy:.4f}, kl_div={kl_div:.4f}"
        )
        if abort_on_collapse:
            raise RuntimeError("DINO collapse detected — aborting to save compute.")
    # Return all metrics for wandb logging
    return {
        "debug/cosine_similarity": cosine_sim,
        "debug/teacher_entropy": entropy,
        "debug/teacher_max_prob": max_prob,
        "debug/kl_divergence": kl_div,
        "debug/student_std_mean": std_mean,
        "debug/student_std_min": std_min,
        "debug/student_std_max": std_max,
        "debug/student_embedding_variance": embedding_var,
        "debug/global_view_diff": view_diff,
        "debug/center_norm": center_norm,
        "debug/delta_center_norm": delta_center,
    }