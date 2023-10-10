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
import PIL.ImageDraw
import cv2
from .owl_predictor import OwlDecodeOutput
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def get_colors(count: int):
    cmap = plt.cm.get_cmap("rainbow", count)
    colors = []
    for i in range(count):
        color = cmap(i)
        color = [int(255 * value) for value in color]
        colors.append(tuple(color))
    return colors


def draw_owl_output(image, output: OwlDecodeOutput, text: List[str], draw_text=True):
    is_pil = not isinstance(image, np.ndarray)
    if is_pil:
        image = np.asarray(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    colors = get_colors(len(text))
    num_detections = len(output.labels)

    for i in range(num_detections):
        box = output.boxes[i]
        label_index = int(output.labels[i])
        box = [int(x) for x in box]
        pt0 = (box[0], box[1])
        pt1 = (box[2], box[3])
        cv2.rectangle(
            image,
            pt0,
            pt1,
            colors[label_index],
            4
        )
        if draw_text:
            offset_y = 12
            offset_x = 0
            label_text = text[label_index]
            cv2.putText(
                image,
                label_text,
                (box[0] + offset_x, box[1] + offset_y),
                font,
                font_scale,
                colors[label_index],
                2,# thickness
                cv2.LINE_AA
            )
    if is_pil:
        image = PIL.Image.fromarray(image)
    return image