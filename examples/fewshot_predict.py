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
import os.path
import time

import PIL.Image
import torch
from nanoowl.fewshot_predictor import FewshotPredictor
from nanoowl.owl_drawing import draw_owl_output
from nanoowl.owl_predictor import OwlPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../assets/cat_image.jpg")
    parser.add_argument(
        "--query-image",
        metavar="N",
        type=str,
        nargs="+",
        help="an example of what to look for in the image",
        default=["../assets/frog.jpg", "../assets/cat_query_image.jpg"],
    )
    parser.add_argument(
        "--query-label",
        metavar="N",
        type=str,
        nargs="+",
        help="a text label for each query image",
        default=["a frog", "a cat"],
    )
    parser.add_argument("--threshold", type=str, default="0.1,0.7")
    parser.add_argument("--output", type=str, default="../data/fewshot_predict_out.jpg")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument(
        "--image_encoder_engine",
        type=str,
        default="../data/owl_image_encoder_patch32.engine",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num_profiling_runs", type=int, default=30)
    args = parser.parse_args()

    image = PIL.Image.open(args.image)

    query_images = []
    for image_file in args.query_image:
        if not os.path.isfile(image_file):
            raise FileNotFoundError(f"File missing from {os.path.abspath(image_file)}")
        else:
            query_images.append(PIL.Image.open(image_file))

    query_labels = args.query_label

    thresholds = args.threshold.strip("][()")
    thresholds = thresholds.split(",")
    if len(thresholds) == 1:
        thresholds = float(thresholds[0])
    else:
        thresholds = [float(x) for x in thresholds]

    engine_path = (
        args.image_encoder_engine if os.path.isfile(args.image_encoder_engine) else None
    )
    if not os.path.isfile(args.image_encoder_engine):
        print(
            f"No image encoder engine found at",
            "{os.path.abspath(args.image_encoder_engine)}.",
            "Continuing without tensorrt...",
        )

    predictor = FewshotPredictor(
        owl_predictor=OwlPredictor(args.model, image_encoder_engine=engine_path)
    )

    query_embeddings = [
        predictor.encode_query_image(image=query_image, text_hints=[query_labels])
        for query_image, query_label in zip(query_images, query_labels)
    ]

    output = predictor.predict(image, query_embeddings, threshold=thresholds)

    if args.profile:
        torch.cuda.current_stream().synchronize()
        t0 = time.perf_counter_ns()
        for i in range(args.num_profiling_runs):
            output = predictor.predict(image, query_embeddings, threshold=thresholds)
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter_ns()
        dt = (t1 - t0) / 1e9
        print(f"PROFILING FPS: {args.num_profiling_runs/dt}")

    image = draw_owl_output(image, output, text=query_labels, draw_text=True)

    image.save(args.output)
