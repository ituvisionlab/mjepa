# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F #GU_

from src.models.utils.modules import (
    Block,
    CrossAttention,
    CrossAttentionBlock
)
from src.utils.tensors import trunc_normal_


class AttentivePooler(nn.Module):
    """ Attentive Pooler """
    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer)
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=False,
                    norm_layer=norm_layer)
                for i in range(depth-1)])

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        q = self.query_tokens.repeat(len(x), 1, 1) #x: [B, num_tokens, D]
        # Compute cross-attention scores
        attn_mask = None
        if mask is not None:
            # token_mask: [B, tokens], shape attn_mask [B, num_queries, tokens]
            attn_mask = mask.unsqueeze(1).expand(-1, q.size(1), -1).unsqueeze(1)  # [B,1,num_queries,tokens]
            
        q = self.cross_attention_block(q, x, attn_mask=attn_mask)

        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q


class AttentiveClassifier(nn.Module):
    """ Attentive Classifier """
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=2, #1000,
        dropout=None,
        complete_block=True,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        )
        # self.linear = nn.Linear(embed_dim, num_classes, bias=True) #GU_
        self.linear1 = nn.Linear(embed_dim, embed_dim//4, bias=True)
        self.linear2 = nn.Linear(embed_dim//4, embed_dim//8, bias=True)
        self.linear3 = nn.Linear(embed_dim//8, num_classes, bias=True)

        # Initialize linear layers
        trunc_normal_(self.linear1.weight, std=init_std)
        trunc_normal_(self.linear2.weight, std=init_std)
        trunc_normal_(self.linear3.weight, std=init_std)
        
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
        nn.init.constant_(self.linear3.bias, 0)

        self.drop_head = None
        if dropout is not None:
            self.drop_head = nn.Dropout(dropout)


    def forward(self, x, contrast_mask=None):
        
        # x: [B, num_tokens, D]
        # contrast_mask: [B, C], to be expanded to [B, tokens] (matching x)

        mask = None
        if contrast_mask is not None:
            # Repeat contrast mask along tokens: expanded contrastMask to mask [B, num_tokens]
            B, num_tokens, _ = x.shape
            C = contrast_mask.shape[1]
            tokens_per_contrast = num_tokens // C
            mask = contrast_mask.bool().unsqueeze(-1).repeat(1, 1, tokens_per_contrast).view(B, num_tokens) #tensor of B,NxC

        x = self.pooler(x, mask=mask).squeeze(1)
        
        # if dropout layer is defined
        if self.drop_head is not None:
            x = self.drop_head(x)

        # x = self.linear(x) #GU_
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)

        return x

class LinearClassifier(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_classes=2,
        n_layers=1,
        init_std=0.02):
        
        super().__init__()
        
        if n_layers > 1:
            self.linear1 = nn.Linear(embed_dim*n_layers, embed_dim, bias=True)
            self.linear2 = nn.Linear(embed_dim, embed_dim//4, bias=True)
            self.linear3 = nn.Linear(embed_dim//4, num_classes, bias=True)
        else:
            self.linear1 = nn.Linear(embed_dim, embed_dim//4, bias=True)
            self.linear2 = nn.Linear(embed_dim//4, embed_dim//8, bias=True)
            self.linear3 = nn.Linear(embed_dim//8, num_classes, bias=True)

        trunc_normal_(self.linear1.weight, std=init_std)
        trunc_normal_(self.linear2.weight, std=init_std)
        trunc_normal_(self.linear3.weight, std=init_std)
        
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
        nn.init.constant_(self.linear3.bias, 0)
            
        self.drop_head = nn.Dropout(0.25)

        
    def forward(self, x):
        
        
        # x = self.drop_head(x)
        
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)

        return x
    
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=2) #softmax over contrast dimension

    def forward(self, embeddings, contrast_mask=None):
        """
        embeddings: list of [B, N, D] from each contrast (C elements)
        contrast_mask: [B, C], 1 for available, 0 for missing contrasts
        returns: [B, N, D] pooled embedding
        """
        x = torch.stack(embeddings, dim=1)  # [B, C, N, D]
        B, C, N, D = x.shape

        contrast_mask = contrast_mask[:, :C]  # Trim to match current contrasts present 
        #b/c contrast_mask might sometimes have fewer than C_max contrasts (e.g., [B,4] instead of [B,6]) 
        # as the pipeline skips contrasts entirely when no sample has them in a batch.

        query = self.query.expand(B, 1, D)  # [B, 1, D]
        keys = self.key_proj(x)             # [B, C, N, D]
        values = self.value_proj(x)         # [B, C, N, D]

        scores = torch.einsum("bqd,bknd->bqkn", query, keys)  # [B, 1, C, N]

        # Masking invalid contrasts: ensure mask dimensions match scores dimensions
        if contrast_mask is not None:
            expanded_mask = contrast_mask.unsqueeze(-1).unsqueeze(1).expand_as(scores)  # [B, 1, C, N]
            scores.masked_fill_(expanded_mask == 0, float('-inf'))                   
            scores = scores.clamp(min=-1e4) # Stability fix (avoid potential overflow in fp16)
 
        attn_weights = self.softmax(scores)                   # [B, 1, C, N]
    
        # DEBUG PRINT: Mean attention over tokens for first sample
        print("Attention weights first sample (mean over tokens):", attn_weights[0, 0].mean(dim=-1).detach())
        print("Contrast mask for first sample:", contrast_mask[0])

        values = values.unsqueeze(1)                          # [B, 1, C, N, D]
        weighted = attn_weights.unsqueeze(-1) * values        # [B, 1, C, N, D]
        pooled = weighted.sum(dim=2)                 # [B, 1, N, D]

        return pooled.squeeze(1)                              # [B, N, D]

