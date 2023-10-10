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
from nanoowl.owl_predictor import (
    OwlPredictor
)
from nanoowl.owl_drawing import (
    draw_owl_output
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../assets/owl_glove_small.jpg")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="../data/owl_glove_out.jpg")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="../data/owlvit_image_encoder_patch32.engine")
    args = parser.parse_args()

    text = args.prompt.split(',')
    print(text)

    predictor = OwlPredictor(
        args.model,
        image_encoder_engine=args.image_encoder_engine
    )

    image = PIL.Image.open(args.image)
    
    text_encodings = predictor.encode_text(text)

    output = predictor.predict(
        image=image, 
        text=text, 
        text_encodings=text_encodings,
        threshold=args.threshold,
        pad_square=False
    )

    print(output)
    image = draw_owl_output(image, output, text=text, draw_text=True)

    image.save(args.output)