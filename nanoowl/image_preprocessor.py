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
import torchvision
import PIL.Image
import numpy as np
from typing import Tuple, Optional, Union


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
            std: Tuple[float, float, float] = DEFAULT_IMAGE_PREPROCESSOR_STD,
            resize: Optional[Union[int, Tuple[int, int]]] = None,
            resize_by_pad: bool = False,
            padding_value: Optional[float] = 127.5,
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

        if resize is not None and isinstance(resize, int):
            resize = (resize, resize)
        self.resize = resize
        self.resize_by_pad = resize_by_pad
        self.padding_value = padding_value
        if (resize is not None) and (not resize_by_pad):
            self.resizer = torchvision.transforms.Resize(
                resize, 
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            )
        else:
            self.resizer = None

    def forward(self, image: torch.Tensor, inplace: bool = False):
        
        if self.resize:
            if self.resizer is not None:
                image = self.resizer(image)
            if self.resize_by_pad:
                if image.size(-1) <= self.resize[-1] and image.size(-2) <= self.resize[-2]:
                    image = torch.nn.functional.pad(
                        image, 
                        [0, self.resize[-1] - image.size(-1), 0, self.resize[-2] - image.size(-2)],
                        "constant",
                        self.padding_value
                    )
                else:
                    downsample_factor = max(image.size(-2) / self.resize[-2], image.size(-1) / self.resize[-1])
                    target_size = (round(image.size(-2) / downsample_factor), round(image.size(-1) / downsample_factor))
                    image = torchvision.transforms.functional.resize(
                        image,
                        target_size,
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                    )
                    image = torch.nn.functional.pad(
                        image, 
                        [0, self.resize[-1] - image.size(-1), 0, self.resize[-2] - image.size(-2)],
                        "constant",
                        self.padding_value
                    )
        
        if inplace:
            image = image.sub_(self.mean).div_(self.std)
        else:
            image = (image - self.mean) / self.std

        return image
    
    @torch.no_grad()
    def preprocess_numpy_array(self, image: np.ndarray):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)[None, ...]
        image = image.to(self.mean.device)
        image = image.type(self.mean.dtype)
        return self.forward(image, inplace=True)

    @torch.no_grad()
    def preprocess_pil_image(self, image: PIL.Image.Image):
        return self.preprocess_numpy_array(np.asarray(image))