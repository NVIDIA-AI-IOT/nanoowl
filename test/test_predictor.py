import pytest
import PIL.Image
from nanoowl.utils.predictor import (
    OwlVitPredictor,
    OwlVitImagePreprocessorModule,
    OwlVitImageEncoderModule
)
from nanoowl.utils.drawing import draw_detections


def test_owlvit_predictor_cuda():

    predictor = OwlVitPredictor.from_pretrained("google/owlvit-base-patch32")

    image = PIL.Image.open("assets/owl_glove.jpg")

    detections = predictor.predict(image=image, text=["an owl", "a glove"])

    draw_detections(image, detections)

    image.save("data/test_owlvit_predictor_cuda.jpg")

