import pytest
import PIL.Image
from nanoowl.model import (
    OwlVitPredictor,
    OwlVitImagePreprocessorModule,
    OwlVitImageEncoderModule
)
from nanoowl.utils.drawing import draw_detections_raw


def test_owlvit_predictor_cuda():

    predictor = OwlVitPredictor.from_hf_pretrained("google/owlvit-base-patch32")

    image = PIL.Image.open("assets/owl_glove.jpg")

    detections = predictor.predict(image=image, text=["an owl", "a glove"])

    draw_detections_raw(image, detections)
    # print(len(detections))
    # for detection in detections:
    #     draw_detection(image, detection)

    image.save("data/test_owlvit_predictor_cuda.jpg")

