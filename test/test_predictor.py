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

    detections = predictor.predict(image, texts=["an owl"])

    for detection in detections:
        draw_detection(image, detection)

    image.save("data/test_predictor_encoder_image.jpg")