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
import clip
import PIL.Image
from torchvision.ops import roi_align
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .image_preprocessor import ImagePreprocessor


__all__ = [
    "ClipPredictor",
    "ClipEncodeTextOutput",
    "ClipEncodeImageOutput",
    "ClipDecodeOutput"
]


@dataclass
class ClipEncodeTextOutput:
    text_embeds: torch.Tensor

    def slice(self, start_index, end_index):
        return ClipEncodeTextOutput(
            text_embeds=self.text_embeds[start_index:end_index]
        )


@dataclass
class ClipEncodeImageOutput:
    image_embeds: torch.Tensor


@dataclass
class ClipDecodeOutput:
    labels: torch.Tensor
    scores: torch.Tensor


class ClipPredictor(torch.nn.Module):
    
    def __init__(self,
            model_name: str = "ViT-B/32",
            image_size: Tuple[int, int] = (224, 224),
            device: str = "cuda",
            image_preprocessor: Optional[ImagePreprocessor] = None
        ):
        super().__init__()
        self.device = device
        self.clip_model, _ = clip.load(model_name, device)
        self.image_size = image_size
        self.mesh_grid = torch.stack(
            torch.meshgrid(
                torch.linspace(0., 1., self.image_size[1]),
                torch.linspace(0., 1., self.image_size[0])
            )
        ).to(self.device).float()
        self.image_preprocessor = image_preprocessor.to(self.device).eval() if image_preprocessor else ImagePreprocessor().to(self.device).eval()
    
    def get_device(self):
        return self.device
        
    def get_image_size(self):
        return self.image_size

    def encode_text(self, text: List[str]) -> ClipEncodeTextOutput:
        text_tokens = clip.tokenize(text).to(self.device)
        text_embeds = self.clip_model.encode_text(text_tokens)
        return ClipEncodeTextOutput(text_embeds=text_embeds)

    def encode_image(self, image: torch.Tensor) -> ClipEncodeImageOutput:
        image_embeds = self.clip_model.encode_image(image)
        return ClipEncodeImageOutput(image_embeds=image_embeds)

    def extract_rois(self, image: torch.Tensor, rois: torch.Tensor, pad_square: bool = True, padding_scale: float=1.0):
        if len(rois) == 0:
            return torch.empty(
                (0, image.shape[1], self.image_size[0], self.image_size[1]),
                dtype=image.dtype,
                device=image.device
            )

        if pad_square:
            # pad square
            w = padding_scale * (rois[..., 2] - rois[..., 0]) / 2
            h = padding_scale * (rois[..., 3] - rois[..., 1]) / 2
            cx = (rois[..., 0] + rois[..., 2]) / 2
            cy = (rois[..., 1] + rois[..., 3]) / 2
            s = torch.max(w, h)
            rois = torch.stack([cx-s, cy-s, cx+s, cy+s], dim=-1)

            # compute mask
            pad_x = (s - w) / (2 * s)
            pad_y = (s - h) / (2 * s)
            mask_x = (self.mesh_grid[1][None, ...] > pad_x[..., None, None]) & (self.mesh_grid[1][None, ...] < (1. - pad_x[..., None, None]))
            mask_y = (self.mesh_grid[0][None, ...] > pad_y[..., None, None]) & (self.mesh_grid[0][None, ...] < (1. - pad_y[..., None, None]))
            mask = (mask_x & mask_y)

        roi_images = roi_align(image, [rois], output_size=self.get_image_size())

        if pad_square:
            roi_images = roi_images * mask[:, None, :, :]

        return roi_images, rois

    def encode_rois(self, image: torch.Tensor, rois: torch.Tensor, pad_square: bool = True, padding_scale: float = 1.0):
        roi_images, rois = self.extract_rois(image, rois, pad_square, padding_scale)
        return self.encode_image(roi_images)

    def decode(self, 
            image_output: ClipEncodeImageOutput, 
            text_output: ClipEncodeTextOutput
        ) -> ClipDecodeOutput:

        image_embeds = image_output.image_embeds
        text_embeds = text_output.text_embeds

        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        probs = torch.softmax(logits_per_image, dim=-1)
        prob_max = probs.max(dim=-1)
        
        return ClipDecodeOutput(
            labels=prob_max.indices,
            scores=prob_max.values
        )

    def predict(self, 
            image: PIL.Image, 
            text: List[str], 
            text_encodings: Optional[ClipEncodeTextOutput],
            pad_square: bool = True,
            threshold: float = 0.1
        ) -> ClipDecodeOutput:

        image_tensor = self.image_preprocessor.preprocess_pil_image(image)

        if text_encodings is None:
            text_encodings = self.encode_text(text)

        rois = torch.tensor([[0, 0, image.height, image.width]], dtype=image_tensor.dtype, device=image_tensor.device)

        image_encodings = self.encode_rois(image_tensor, rois, pad_square=pad_square)

        return self.decode(image_encodings, text_encodings)