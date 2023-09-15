import pytest
import os
import PIL.Image

from nanoowl.utils.drawing import draw_detection
from nanoowl.utils.predictor import Predictor


def test_predictor_encoder_image():

    predictor = Predictor(
        threshold=0.1, 
    )

    image = PIL.Image.open("assets/owl_glove.jpg")

    detections = predictor.predict_text(image=image, text=["an owl", "a glove"])

    for detection in detections:
        draw_detection(image, detection)

    image.save("data/test_predictor_encoder_image.jpg")

def test_predictor_pre_set_text():

    predictor = Predictor(
        threshold=0.1, 
    )

    image = PIL.Image.open("assets/owl_glove.jpg")

    predictor.set_text(["an owl", "a glove"])

    detections = predictor.predict_text(image=image)

    for detection in detections:
        draw_detection(image, detection)

    image.save("data/test_predictor_pre_set_text.jpg")


def test_predictor_pre_set_image():

    predictor = Predictor(
        threshold=0.1, 
    )

    image = PIL.Image.open("assets/owl_glove.jpg")

    predictor.set_image(image)

    detections = predictor.predict_text(text=["an owl", "a glove"])

    for detection in detections:
        draw_detection(image, detection)

    image.save("data/test_predictor_pre_set_image.jpg")


def test_predictor_query_image():
    import requests
    import numpy as np

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    query_image_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
    image = PIL.Image.open(requests.get(image_url, stream=True).raw)
    query_image = PIL.Image.open(requests.get(query_image_url, stream=True).raw)

    # image = PIL.Image.open("assets/owl_glove.jpg")

    # query_image = PIL.Image.fromarray(np.zeros_like(np.asarray(image)))

    # box = [1000, 300, 2000, 1600]
    # query_image_crop = image.crop([1000, 300, 2000, 1600])
    # query_image.paste(query_image_crop, box)

    # image = image.resize((640, 480))
    # query_image = query_image.resize((640, 480))

    predictor = Predictor(
        threshold=0.6, 
        query_image_nms_threshold=0.3,
        patch_size=16
    )

    detections = predictor.predict_query_image(image, query_image)

    for detection in detections:
        draw_detection(image, detection, draw_text=False)

    image.save("data/test_predictor_query_image.jpg")
