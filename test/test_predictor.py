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

    detections = predictor.predict(image=image, text=["an owl", "a glove"])

    for detection in detections:
        draw_detection(image, detection)

    image.save("data/test_predictor_encoder_image.jpg")

def test_predictor_pre_set_text():

    predictor = Predictor(
        threshold=0.1, 
    )

    image = PIL.Image.open("assets/owl_glove.jpg")

    predictor.set_text(["an owl", "a glove"])

    detections = predictor.predict(image=image)

    for detection in detections:
        draw_detection(image, detection)

    image.save("data/test_predictor_pre_set_text.jpg")


def test_predictor_pre_set_image():

    predictor = Predictor(
        threshold=0.1, 
    )

    image = PIL.Image.open("assets/owl_glove.jpg")

    predictor.set_image(image)

    detections = predictor.predict(text=["an owl", "a glove"])

    for detection in detections:
        draw_detection(image, detection)

    image.save("data/test_predictor_pre_set_image.jpg")