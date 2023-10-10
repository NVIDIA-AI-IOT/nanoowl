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
import PIL.Image
from nanoowl.tree_predictor import TreePredictor
from nanoowl.tree import Tree


def test_encode_clip_labels():
    
    predictor = TreePredictor()
    tree = Tree.from_prompt("(sunny, rainy)")

    text_encodings = predictor.encode_clip_text(tree)

    assert len(text_encodings) == 2
    assert 1 in text_encodings
    assert 2 in text_encodings
    assert text_encodings[1].text_embeds.shape == (1, 512)


def test_encode_owl_labels():
    
    predictor = TreePredictor()
    tree = Tree.from_prompt("[a face [an eye, a nose]]")

    text_encodings = predictor.encode_owl_text(tree)

    assert len(text_encodings) == 3
    assert 1 in text_encodings
    assert 2 in text_encodings
    assert 3 in text_encodings
    assert text_encodings[1].text_embeds.shape == (1, 512)


def test_encode_clip_owl_labels_mixed():
    
    predictor = TreePredictor()
    tree = Tree.from_prompt("[a face [an eye, a nose](happy, sad)]")

    owl_text_encodings = predictor.encode_owl_text(tree)
    clip_text_encodings = predictor.encode_clip_text(tree)

    assert len(owl_text_encodings) == 3
    assert len(clip_text_encodings) == 2


def test_tree_predictor_predict():

    predictor = TreePredictor()
    tree = Tree.from_prompt("[an owl]")


    image = PIL.Image.open("assets/owl_glove.jpg")

    detections = predictor.predict(image, tree)

    
def test_tree_predictor_predict():

    predictor = TreePredictor()
    tree = Tree.from_prompt("(outdoors, indoors)")


    image = PIL.Image.open("assets/owl_glove.jpg")

    detections = predictor.predict(image, tree)

    print(detections)
    