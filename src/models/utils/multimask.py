# mjepa: A 3D MRI self-supervised learning framework based on a modified V-JEPA
# Copyright (c) 2024–2025 [Gozde Unal, NYU]
#
# This file is based on an earlier version of code from:
# V-JEPA (https://github.com/facebookresearch/v-jepa)
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This codebase has been significantly modified for use in medical imaging and 3D MRI.
# All modifications are licensed under the original MIT license (or the applicable license).

import torch
import torch.nn as nn
from src.masks.utils import apply_masks

class MultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None, **kwargs):
        if masks is None:
            return self.backbone(x, masks=masks, **kwargs)

        if (masks is not None) and not isinstance(masks, list):
            masks = [masks]

        outs = []
        for m in masks:
            out = self.backbone(x, masks=m, **kwargs)
            outs.append(out)
            # outs.append(self.backbone(x, masks=m, **kwargs))  # outs += [self.backbone(x, masks=m)]


        return outs


class PredictorMultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt):
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(tgt) is not list:
            tgt = [tgt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]

        outs = []
        for i, (zi, hi, mc, mt) in enumerate(zip(ctxt, tgt, masks_ctxt, masks_tgt)):

            assert isinstance(zi, torch.Tensor), f"zi is not tensor: {type(zi)}"
            assert isinstance(hi, torch.Tensor), f"hi is not tensor: {type(hi)}"
            assert zi.ndim == 3, f"zi shape should be [B, N, D], got {zi.shape}"

            outs += [self.backbone(zi, hi, mc, mt, mask_index=i)]
        return outs

class DecoderMultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, ctxt, masks_ctxt, masks_tgt,return_all_tokens=False):
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]

        outs = []
        for i, (zi, mc, mt) in enumerate(zip(ctxt, masks_ctxt, masks_tgt)):
            outs += [self.backbone(zi, mc, mt, mask_index=i,return_all_tokens=return_all_tokens)]
        return outs