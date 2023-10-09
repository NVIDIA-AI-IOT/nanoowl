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
import PIL.Image
import numpy as np
from typing import Tuple


__all__ = [
    "ImagePreprocessor",
    "DEFAULT_IMAGE_PREPROCESSOR_MEAN",
    "DEFAULT_IMAGE_PREPROCESSOR_STD"
]


DEFAULT_IMAGE_PREPROCESSOR_MEAN = [
    0.48145466 * 255., 
    0.4578275 * 255., 
    0.40821073 * 255.
]


DEFAULT_IMAGE_PREPROCESSOR_STD = [
    0.26862954 * 255., 
    0.26130258 * 255., 
    0.27577711 * 255.
]


class ImagePreprocessor(torch.nn.Module):
    def __init__(self,
            mean: Tuple[float, float, float] = DEFAULT_IMAGE_PREPROCESSOR_MEAN,
            std: Tuple[float, float, float] = DEFAULT_IMAGE_PREPROCESSOR_STD
        ):
        super().__init__()
        
        self.register_buffer(
            "mean",
            torch.tensor(mean)[None, :, None, None]
        )
        self.register_buffer(
            "std",
            torch.tensor(std)[None, :, None, None]
        )

    def forward(self, image: torch.Tensor, inplace: bool = False):

        if inplace:
            image = image.sub_(self.mean).div_(self.std)
        else:
            image = (image - self.mean) / self.std

        return image
    
    @torch.no_grad()
    def preprocess_pil_image(self, image: PIL.Image.Image):
        image = torch.from_numpy(np.asarray(image))
        image = image.permute(2, 0, 1)[None, ...]
        image = image.to(self.mean.device)
        image = image.type(self.mean.dtype)
        return self.forward(image, inplace=True)