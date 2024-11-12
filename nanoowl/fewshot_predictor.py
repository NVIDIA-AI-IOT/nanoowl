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


from typing import List, Optional, Union

import PIL.Image
import torch

from .image_preprocessor import ImagePreprocessor
from .owl_predictor import (
    OwlDecodeOutput,
    OwlEncodeImageOutput,
    OwlEncodeTextOutput,
    OwlPredictor,
)


class FewshotPredictor(torch.nn.Module):
    def __init__(
        self,
        owl_predictor: Optional[OwlPredictor] = None,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        device: str = None,
    ):
        super().__init__()
        device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.owl_predictor = (
            OwlPredictor(device=device) if owl_predictor is None else owl_predictor
        )
        self.image_preprocessor = (
            ImagePreprocessor().to(device).eval()
            if image_preprocessor is None
            else image_preprocessor
        )

    @torch.no_grad()
    def predict(
        self,
        image: PIL.Image,
        query_embeddings: List,
        threshold: Union[int, float, List[Union[int, float]]] = 0.1,
        pad_square: bool = True,
    ) -> OwlDecodeOutput:
        image_tensor = self.image_preprocessor.preprocess_pil_image(image)

        rois = torch.tensor(
            [[0, 0, image.width, image.height]],
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        )

        image_encodings = self.owl_predictor.encode_rois(
            image_tensor, rois, pad_square=pad_square
        )

        return self.decode(image_encodings, query_embeddings, threshold)

    def decode(
        self,
        image_output: OwlEncodeImageOutput,
        query_embeds,
        threshold: Union[int, float, List[Union[int, float]]] = 0.1,
    ) -> OwlDecodeOutput:
        num_input_images = image_output.image_class_embeds.shape[0]

        image_class_embeds = image_output.image_class_embeds
        image_class_embeds = image_class_embeds / (
            torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
        )

        if isinstance(threshold, (int, float)):
            threshold = [threshold] * len(
                query_embeds
            )  # apply single threshold to all labels

        query_embeds = torch.concat(query_embeds, dim=0)
        query_embeds = query_embeds / (
            torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
        )
        logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
        logits = (logits + image_output.logit_shift) * image_output.logit_scale

        scores_sigmoid = torch.sigmoid(logits)
        scores_max = scores_sigmoid.max(dim=-1)
        labels = scores_max.indices
        scores = scores_max.values
        masks = []
        for i, thresh in enumerate(threshold):
            label_mask = labels == i
            score_mask = scores > thresh
            obj_mask = torch.logical_and(label_mask, score_mask)
            masks.append(obj_mask)
        mask = masks[0]
        for mask_t in masks[1:]:
            mask = torch.logical_or(mask, mask_t)

        input_indices = torch.arange(
            0, num_input_images, dtype=labels.dtype, device=labels.device
        )
        input_indices = input_indices[:, None].repeat(1, self.owl_predictor.num_patches)

        return OwlDecodeOutput(
            labels=labels[mask],
            scores=scores[mask],
            boxes=image_output.pred_boxes[mask],
            input_indices=input_indices[mask],
        )

    def encode_query_image(
        self,
        image: PIL.Image,
        text_hints: List[str],
        pad_square: bool = True,
    ) -> torch.Tensor:
        image_tensor = self.image_preprocessor.preprocess_pil_image(image)

        text_encodings = self.encode_text(text_hints)

        rois = torch.tensor(
            [[0, 0, image.width, image.height]],
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        )

        image_encodings = self.owl_predictor.encode_rois(
            image_tensor, rois, pad_square=pad_square
        )

        return self.find_best_encoding(image_encodings, text_encodings)

    def encode_text(self, texts: List[str]) -> OwlEncodeTextOutput:
        return self.owl_predictor.encode_text(texts)

    @staticmethod
    def find_best_encoding(
        image_output: OwlEncodeImageOutput,
        text_output: OwlEncodeTextOutput,
    ) -> torch.Tensor:
        image_class_embeds = image_output.image_class_embeds
        image_class_embeds = image_class_embeds / (
            torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
        )
        query_embeds = text_output.text_embeds
        query_embeds = query_embeds / (
            torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
        )
        logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
        logits = (logits + image_output.logit_shift) * image_output.logit_scale

        scores_sigmoid = torch.sigmoid(logits)
        scores_max = scores_sigmoid.max(dim=-1)
        scores = scores_max.values
        best = torch.argmax(scores).item()
        best_embed = image_class_embeds[:, best]
        return best_embed
