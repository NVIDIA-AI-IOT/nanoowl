import pytest
import PIL.Image
import os
from nanoowl.model import (
    OwlVitPredictor
)
from nanoowl.utils.drawing import draw_detections_raw


if __name__ == "__main__":
    
    hf_name = "google/owlvit-base-patch32"
    folder = os.path.join("data", os.path.basename(hf_name))

    if not os.path.exists(folder):
        os.makedirs(folder)

    image_encoder_path = os.path.join(folder, os.path.basename(hf_name) + "-image-encoder.onnx")
    
    predictor = OwlVitPredictor.from_pretrained(hf_name)

    predictor.image_encoder.export_onnx(image_encoder_path)

    # draw_detections_raw(image, detections)
    # # print(len(detections))
    # # for detection in detections:
    # #     draw_detection(image, detection)

    # image.save("data/test_owlvit_predictor_cuda.jpg")

