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
from nanoowl.clip_predictor import ClipPredictor
from nanoowl.image_preprocessor import ImagePreprocessor


def test_get_image_size():
    clip_predictor = ClipPredictor()
    assert clip_predictor.get_image_size() == (224, 224)


def test_clip_encode_text():

    clip_predictor = ClipPredictor()

    text_encode_output = clip_predictor.encode_text(["a frog", "a dog"])

    assert text_encode_output.text_embeds.shape == (2, 512)


def test_clip_encode_image():

    clip_predictor = ClipPredictor()

    image = PIL.Image.open("assets/owl_glove_small.jpg")

    image = image.resize((224, 224))

    image_preprocessor = ImagePreprocessor().to(clip_predictor.device).eval()

    image_tensor = image_preprocessor.preprocess_pil_image(image)

    image_encode_output = clip_predictor.encode_image(image_tensor)

    assert image_encode_output.image_embeds.shape == (1, 512)


def test_clip_classify():

    clip_predictor = ClipPredictor()

    image = PIL.Image.open("assets/frog.jpg")

    image = image.resize((224, 224))
    image_preprocessor = ImagePreprocessor().to(clip_predictor.device).eval()

    image_tensor = image_preprocessor.preprocess_pil_image(image)

    text_output = clip_predictor.encode_text(["a frog", "an owl"])
    image_output = clip_predictor.encode_image(image_tensor)

    classify_output = clip_predictor.decode(image_output, text_output)

    assert classify_output.labels[0] == 0