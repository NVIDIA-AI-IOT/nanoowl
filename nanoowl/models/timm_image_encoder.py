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
from .registry import register_model


class TimmImageEncoder(nn.Module):
    def __init__(self, 
            model_name: str = "resnet18",
            pretrained: bool = False,
            feature_dim: int = 768,
            feature_shape: Tuple[int, int] = (24, 24),
            pos_embedding: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True
        )

        channels = self.backbone.feature_info.channels()

        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, feature_dim, 1, padding=0)
        )

        if pos_embedding:
            self.register_parameter(
                "pos_embedding", 
                nn.Parameter(1e-5*torch.randn(1, feature_dim, *feature_shape))
            )
        else:
            self.pos_embedding = None
        
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x[-1])
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        return x
    

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
def resnet18():
    return TimmImageEncoder('efficientvit_b0', pretrained=True)