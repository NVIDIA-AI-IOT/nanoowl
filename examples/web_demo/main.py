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


import asyncio
from aiohttp import web, WSCloseCode
import logging
import weakref
import cv2
import time
import PIL.Image
import matplotlib.pyplot as plt
from typing import List
from nanoowl.utils.predictor import OwlVitPredictor
from nanoowl.utils.drawing import draw_detections_cv2


CAMERA_DEVICE = 0
IMAGE_QUALITY = 50
MODEL_NAME = "google/owlvit-base-patch32"
IMAGE_ENCODER_ENGINE = "../../data/owlvit-base-patch32-image-encoder.engine"

prompt: str = None


def get_colors(count: int):
    cmap = plt.cm.get_cmap("rainbow", count)
    colors = []
    for i in range(count):
        color = cmap(i)
        color = [int(255 * value) for value in color]
        colors.append(tuple(color))
    return colors


def cv2_to_pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(image)


async def handle_index_get(request: web.Request):
    logging.info("handle_index_get")
    return web.FileResponse("./index.html")


async def websocket_handler(request):

    global prompt

    ws = web.WebSocketResponse()

    await ws.prepare(request)

    logging.info("Websocket connected.")

    request.app['websockets'].add(ws)

    try:
        async for msg in ws:
            logging.info(f"Received message from websocket.")
            if "prompt" in msg.data:
                header, prompt = msg.data.split(":")
                logging.info("Set prompt: " + prompt)
    finally:
        request.app['websockets'].discard(ws)

    return ws


async def on_shutdown(app: web.Application):
    for ws in set(app['websockets']):
        await ws.close(code=WSCloseCode.GOING_AWAY,
                       message='Server shutdown')


async def detection_loop(app: web.Application):

    loop = asyncio.get_running_loop()

    logging.info("Opening camera.")

    camera = cv2.VideoCapture(CAMERA_DEVICE)

    logging.info("Loading predictor.")

    predictor = OwlVitPredictor.from_pretrained(
        MODEL_NAME,
        image_encoder_engine=IMAGE_ENCODER_ENGINE,
        device="cuda"
    )

    def _read_and_encode_image():

        re, image = camera.read()

        if not re:
            return re, None

        image_pil = cv2_to_pil(image)


        if prompt is not None:
            text = [s.strip() for s in prompt.split(',')]
            print(len(text))
            colors = get_colors(len(text))
            detections = predictor.predict(image=image_pil, text=text)
            image = draw_detections_cv2(image, detections, colors=colors)
            print(colors)

        image_jpeg = bytes(
            cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])[1]
        )

        return re, image_jpeg

    while True:

        re, image = await loop.run_in_executor(None, _read_and_encode_image)
        
        if not re:
            break
        
        for ws in app["websockets"]:
            await ws.send_bytes(image)

    camera.release()


async def run_detection_loop(app):
    try:
        task = asyncio.create_task(detection_loop(app))
        yield
        task.cancel()
    except asyncio.CancelledError:
        pass
    finally:
        await task


logging.basicConfig(level=logging.INFO)
app = web.Application()
app['websockets'] = weakref.WeakSet()
app.router.add_get("/", handle_index_get)
app.router.add_route("GET", "/ws", websocket_handler)
app.on_shutdown.append(on_shutdown)
app.cleanup_ctx.append(run_detection_loop)
web.run_app(app, host="0.0.0.0", port=7860)