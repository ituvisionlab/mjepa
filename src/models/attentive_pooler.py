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

    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
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


    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, embeddings):
        """
        embeddings: list of [B, N, D] from each contrast
        returns: [B, N, D] pooled embedding
        """
        x = torch.stack(embeddings, dim=1)  # [B, C, N, D]
        B, C, N, D = x.shape

        query = self.query.expand(B, 1, D)  # [B, 1, D]
        keys = self.key_proj(x)            # [B, C, N, D]
        values = self.value_proj(x)        # [B, C, N, D]

        scores = torch.einsum("bqd,bknd->bqkn", query, keys)  # [B, 1, C, N]
        attn_weights = self.softmax(scores)                   # [B, 1, C, N]

        values = values.unsqueeze(1)                          # [B, 1, C, N, D]
        weighted = attn_weights.unsqueeze(-1) * values        # [B, 1, C, N, D]
        pooled = torch.sum(weighted, dim=2)                   # [B, 1, N, D]

        return pooled.squeeze(1)                              # [B, N, D]

