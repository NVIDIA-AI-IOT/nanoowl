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
import gradio as gr
import numpy as np
from nanoowl.utils.predictor import (
    OwlVitPredictor
)
from nanoowl.utils.drawing import draw_detections


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="data/owlvit-base-patch32-image-encoder.engine")
    # parser.add_argument("--text_encoder_engine", type=str, default="data/owlvit-base-patch32-text-encoder.engine")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.image_encoder_engine.lower() == "none":
        args.image_encoder_engine = None

    # if args.text_encoder_engine.lower() == "none":
    #     args.text_encoder_engine = None
        
    predictor = OwlVitPredictor.from_pretrained(
        args.model,
        image_encoder_engine=args.image_encoder_engine,
        # text_encoder_engine=args.text_encoder_engine,
        device="cuda"
    )

    def infer(image, text, threshold):
        image = PIL.Image.fromarray(image)
        text = [t.strip() for t in text.split(',')]
        print(text)
        detections = predictor.predict(image=image, text=text, threshold=threshold)
        draw_detections(image, detections)
        return np.asarray(image)

    demo = gr.Interface(
        fn=infer, 
        inputs=[
            gr.Image(value="assets/owl_glove_small.jpg"), 
            gr.Text(value="an owl, a glove, a face"), 
            gr.Slider(minimum=0, maximum=1, value=0.1, step=0.0025)
        ], 
        outputs=["image"],
        title=f"TRTOWL ({args.model})"
    )

    demo.launch(
        server_name=args.host,
        server_port=args.port
    )