import pytest
import PIL.Image
from nanoowl.model import (
    OwlVitPredictor,
    OwlVitImagePreprocessorModule,
    OwlVitImageEncoderModule,
    Profiler, Timer
)
from nanoowl.utils.drawing import draw_detections_raw


if __name__ == "__main__":
        
    predictor = OwlVitPredictor.from_hf_pretrained("google/owlvit-base-patch32")

    image = PIL.Image.open("assets/camera.jpg")

    detections = predictor.predict(image=image, text=["an owl", "a glove"])
    
    p = Profiler()

    with p:
        for i in range(20):
            detections = predictor.predict(image=image, text=["an owl", "a glove"])
    
    p.print_median_elapsed_times_ms()

    # draw_detections_raw(image, detections)
    # # print(len(detections))
    # # for detection in detections:
    # #     draw_detection(image, detection)

    # image.save("data/test_owlvit_predictor_cuda.jpg")

