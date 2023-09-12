import PIL.Image
import matplotlib.pyplot as plt
from nanoowl.utils.predictor import Predictor
from nanoowl.utils.drawing import draw_detection
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="assets/owl_glove.jpg")
    parser.add_argument("--vision_engine", type=str, default="data/owlvit_vision_model.engine")
    parser.add_argument("--prompt", nargs='+', required=True)
    parser.add_argument("--thresh", type=float, default=0.1)
    args = parser.parse_args()
    
    predictor = Predictor(threshold=args.thresh, vision_engine=args.vision_engine)

    image = PIL.Image.open(args.image)

    detections = predictor.predict(image, texts=["an owl", "a glove", "a face"])

    for detection in detections:
        draw_detection(image, detection)

    image.save("data/basic_usage_out.jpg")