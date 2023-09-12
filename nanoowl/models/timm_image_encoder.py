# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from typing import Tuple
import timm
import math
from .registry import register_model
from torch import Tensor


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
    

class TimmImageEncoder(nn.Module):
    def __init__(self, 
            model_name: str = "resnet18",
            pretrained: bool = False,
            embed_dim: int = 768,
            feature_shape: Tuple[int, int] = (24, 24),
            num_attn_heads=1,
            mlp_hidden_size=None,
            mlp_act=nn.GELU,
            include_post_layernorm=False
        ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True
        )

        channels = self.backbone.feature_info.channels()[-1]

        # Map vision features to embedding dimension
        self.feature_proj = nn.Sequential(
            nn.Conv2d(channels, embed_dim, 1, padding=0)
        )

        # Apply position embedding to vision features
        self.register_parameter(
            "pos_embedding", 
            nn.Parameter(1e-5 * torch.randn(1, feature_shape[0] * feature_shape[1], embed_dim))
        )

        # Concatenate special token at index 0 with embeddings
        self.register_parameter(
            "special_token",
            nn.Parameter(1e-5 * torch.randn(1, 1, embed_dim))
        )

        # Apply one multi-head attention layer
        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads=num_attn_heads
        )
        
        # Apply MLP after attention layer
        self.pre_proj_norm = nn.LayerNorm(embed_dim)

        if mlp_hidden_size is None:
            mlp_hidden_size = embed_dim

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_size),
            mlp_act(),
            nn.Linear(mlp_hidden_size, embed_dim)
        )

        # Apply post layer-norm to pooled output
        self.include_post_layernorm = include_post_layernorm
        if self.include_post_layernorm:
            self.post_layernorm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):

        # Flatten vision features
        features = self.backbone(x)[-1]
        features = self.feature_proj(features)
        b, c, h, w = features.shape
        features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Add learned position embedding to vision features
        features = features + self.pos_embedding

        # Concatenate features with special token
        special_token = self.special_token.repeat((b, 1, 1))
        embeddings = torch.cat([special_token, features], dim=1)

        # Apply MHA
        res = embeddings
        embeddings = self.pre_attn_norm(embeddings)
        embeddings = res + self.attn(embeddings, embeddings, embeddings)

        # Apply MLP
        res = embeddings
        embeddings = self.pre_proj_norm(embeddings)
        embeddings = res + self.proj(embeddings)

        # Pooled output
        last_hidden_state = embeddings
        pooled_output = last_hidden_state[:, 0, :]

        if self.include_post_layernorm:
            pooled_output = self.post_layernorm(pooled_output)

        return last_hidden_state, pooled_output
    

@register_model("resnet18")
def resnet18():
    return TimmImageEncoder('resnet18', pretrained=True)


@register_model("resnet34")
def resnet34():
    return TimmImageEncoder('resnet34', pretrained=True)


@register_model("resnet50")
def resnet50():
    return TimmImageEncoder('resnet50', pretrained=True)


@register_model("efficientvit_b0")
def efficientvit_b0():
    return TimmImageEncoder('efficientvit_b0', pretrained=True)


@register_model("efficientvit_b0_h8")
def efficientvit_b0_h8():
    return TimmImageEncoder('efficientvit_b0', pretrained=True, num_attn_heads=8)