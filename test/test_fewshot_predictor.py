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


import PIL.Image
from nanoowl.fewshot_predictor import FewshotPredictor


def test_encode_query_images():
    predictor = FewshotPredictor(device="cpu")

    query_image = PIL.Image.open("assets/frog.jpg")

    query_encoding = predictor.encode_query_image(query_image, ["a frog"])

    assert len(query_encoding.shape) == 2
    assert query_encoding.shape[0] == 1
    assert query_encoding.shape[1] == 512


def test_encode_labels():
    predictor = FewshotPredictor()

    labels = ["a frog", "an owl", "mice", "405943069245", ""]

    text_encodings = predictor.encode_text(labels).text_embeds

    assert len(text_encodings.shape) == 2
    assert text_encodings.shape[0] == len(labels)
    assert text_encodings.shape[1] == 512


def test_fewshot_predictor_predict():
    predictor = FewshotPredictor()

    image = PIL.Image.open("../assets/cat_query_image.jpg")

    query_image = PIL.Image.open("../assets/cat_image.jpg")

    query_label = "a cat"

    thresholds = 0.7

    query_embedding = predictor.encode_query_image(
        image=query_image, text_hints=[query_label]
    )

    detections = predictor.predict(image, [query_embedding], threshold=thresholds)

    print(detections)
