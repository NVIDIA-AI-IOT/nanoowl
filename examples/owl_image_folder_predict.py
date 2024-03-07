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


import argparse
import PIL.Image
import time
import os
import torch
from nanoowl.owl_predictor import (
    OwlPredictor
)
from nanoowl.owl_drawing import (
    draw_owl_output
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="../../example_imgs_openx/")
    parser.add_argument("--prompt", type=str, default="[eggplant,microwave,banana,fork,yellow towel,red towel,blue towel,red bowl,blue bowl,purple towel,steel bowl,white bowl,red spoon,green spoon,blue spoon,can,strawberry,corn,yellow plate,red plate,cabinet,fridge,screwdriver,mushroom,plastic bottle,green chip bag,brown chip bag,blue chip bag,apple,orange]")
    parser.add_argument("--threshold", type=str, default="0.1,0.1")
    parser.add_argument("--nms_threshold", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="../data/example_imgs_openx_prediction/")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument('--no_roi_align', action='store_true')
    parser.add_argument("--image_encoder_engine", type=str, default="../data/owl_image_encoder_patch32.engine")
    args = parser.parse_args()

    prompt = args.prompt.strip("][()")
    text = prompt.split(',')
    print(text)

    thresholds = args.threshold.strip("][()")
    thresholds = thresholds.split(',')
    if len(thresholds) == 1:
        thresholds = float(thresholds[0])
    else:
        thresholds = [float(x) for x in thresholds]
    print(thresholds)
    

    predictor = OwlPredictor(
        args.model,
        image_encoder_engine=args.image_encoder_engine,
        no_roi_align=args.no_roi_align
    )

    text_encodings = predictor.encode_text(text)
    os.makedirs(args.output_dir, exist_ok=True)

    for image_path in os.listdir(args.image_dir):
        image = PIL.Image.open(os.path.join(args.image_dir, image_path))
        
        output = predictor.predict(
            image=image, 
            text=text, 
            text_encodings=text_encodings,
            threshold=thresholds,
            nms_threshold=args.nms_threshold,
            pad_square=False
        )

        image = draw_owl_output(image, output, text=text, draw_text=True)

        image.save(os.path.join(args.output_dir, image_path))