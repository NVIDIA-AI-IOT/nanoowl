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
from nanoowl.utils.predictor import (
    OwlVitPredictor
)
from nanoowl.utils.drawing import draw_detections


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="assets/owl_glove.jpg")
    parser.add_argument("--text", action='append', required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="data/owl_glove_out.jpg")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="data/owlvit-base-patch32-image-encoder.engine")
    parser.add_argument("--output_path", type=str, default="data/predict_out.jpg")
    args = parser.parse_args()


    predictor = OwlVitPredictor.from_pretrained(
        args.model,
        image_encoder_engine=args.image_encoder_engine,
        device="cuda"
    )

    image = PIL.Image.open(args.image)

    text = args.text

    detections = predictor.predict(image=image, text=text, threshold=args.threshold)

    draw_detections(image, detections)

    image.save(args.output)