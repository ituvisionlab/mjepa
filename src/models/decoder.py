# mjepa: A 3D MRI self-supervised learning framework based on a modified V-JEPA
# Copyright (c) 2024–2025 [Gozde Unal, NYU]
#
# This file is based on an earlier version of code from:
# V-JEPA (https://github.com/facebookresearch/v-jepa)
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This codebase has been significantly modified for use in medical imaging and 3D MRI.
# All modifications are licensed under the original MIT license (or the applicable license).

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from src.masks.utils import apply_masks


class VisionTransformerDecoder(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=1,
        zero_init_mask_tokens=True,
        in_chans=3,
        **kwargs
    ):
        super().__init__()
        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        
        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            for i in range(num_mask_tokens)
        ])

        # Determine positional embedding
        self.input_size = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size

        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size)
                * (img_size // patch_size)
                * (img_size // patch_size)
            )
        else:
            self.num_patches = num_patches = (
                (img_size // patch_size)
                * (img_size // patch_size)
            )
        # Position embedding
        self.uniform_power = uniform_power
        self.predictor_pos_embed = None
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim),
            requires_grad=False)

        # Attention Blocks
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                attn_drop=attn_drop_rate,
                grid_size=grid_size,
                grid_depth=grid_depth,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, tubelet_size*(patch_size**2) * in_chans, bias=True)

        #GU_DEBUG: May30
        # print("proj out shape", self.predictor_proj.weight.shape)
        # print("expected output dim", self.tubelet_size * self.patch_size**2 * in_chans)

        # ------ initialize weights
        if self.predictor_pos_embed is not None:
            self._init_pos_embed(self.predictor_pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, ctxt, masks_ctxt, masks_tgt, mask_index=1, return_all_tokens=False):
        """
        :param ctxt: context tokens
        :param masks_ctxt: indices of context tokens in input
        :params masks_tgt: indices of target tokens in input
        """
        assert (masks_ctxt is not None) and (masks_tgt is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]

        if isinstance(masks_tgt, torch.Tensor):
            masks_tgt = [masks_tgt]

        # Batch Size
        B = len(ctxt) // len(masks_ctxt)
        #print(f"[DEBUG] ctxt shape: {ctxt.shape}, Batch size: {B}")

        # Map context tokens to decoder dimensions
        x = self.predictor_embed(ctxt)
        _, N_ctxt, D = x.shape
        N_tgts = masks_tgt[0].shape[1]

        #print(f"[DEBUG] Context tokens: N_ctxt={N_ctxt}, Target tokens: N_tgts={N_tgts}, Embed dim={D}")

        # Add positional embedding to ctxt tokens
        if self.predictor_pos_embed is not None:
            ctxt_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(ctxt_pos_embed, masks_ctxt)

        # Mask token injection
        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index]  # [1, 1, D]
        pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_tgt)

        if self.predictor_pos_embed is not None:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_tgt)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt))
            pred_tokens += pos_embs

        # Concatenate context and masked tokens
        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        expected_tokens = N_ctxt + N_tgts
        assert x.shape[1] == expected_tokens, (
            f"[ERROR] Token count mismatch in decoder: {x.shape[1]} vs {expected_tokens}"
        )
        #print(f"[DEBUG] Total decoder input tokens: {x.shape[1]}")

        # Merge masks (currently assuming 1:1)
        masks_ctxt = torch.cat(masks_ctxt, dim=0)
        masks_tgt = torch.cat(masks_tgt, dim=0)
        masks = torch.cat([masks_ctxt, masks_tgt], dim=1)


        # Transformer layers
        for blk in self.predictor_blocks:
            x = blk(x, mask=masks)
        x = self.predictor_norm(x)

        assert x.shape[1] == expected_tokens, (
            f"[ERROR] Output token count mismatch after transformer: {x.shape[1]} vs {expected_tokens}"
        )

        # Output processing
        if not return_all_tokens:
            x_target = x[:, N_ctxt:]
            x_proj = self.predictor_proj(x_target)
            # print(f"[DEBUG] Returned target prediction shape: {x_proj.shape}")
            return x_proj
        else:
            x_context = x[:, :N_ctxt]
            x_target = x[:, N_ctxt:N_ctxt + N_tgts]
            x_context = self.predictor_proj(x_context)
            x_target = self.predictor_proj(x_target)
            # print(f"[DEBUG] Returned full prediction: context {x_context.shape}, target {x_target.shape}")
            return x_context, x_target


def vit_decoder(**kwargs):
    model = VisionTransformerDecoder(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model

def vit_decoder_light(**kwargs):
    model = VisionTransformerDecoder(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model

def vit_decoder_mod(**kwargs): #input num_heads=4
    model = VisionTransformerDecoder(
        mlp_ratio=2, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model