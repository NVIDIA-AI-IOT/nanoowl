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


import pytest
import torch
import PIL.Image
from nanoowl.image_preprocessor import ImagePreprocessor


def test_image_preprocessor_preprocess_pil_image():

    image_preproc = ImagePreprocessor().to("cuda").eval()

    image = PIL.Image.open("assets/owl_glove_small.jpg")

    image_tensor = image_preproc.preprocess_pil_image(image)

    assert image_tensor.shape == (1, 3, 499, 756)
    assert torch.allclose(image_tensor.mean(), torch.zeros_like(image_tensor), atol=1, rtol=1)
