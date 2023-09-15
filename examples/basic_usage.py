import PIL.Image
import matplotlib.pyplot as plt
from nanoowl.utils.predictor import Predictor
from nanoowl.utils.drawing import draw_detection
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="assets/owl_glove.jpg")
    parser.add_argument("--vision_engine", type=str, default=None)
    parser.add_argument("--vision_model_name", type=str, default=None)
    parser.add_argument("--vision_checkpoint", type=str, default=None)
    parser.add_argument("--prompt", action='append', required=True)
    parser.add_argument("--thresh", type=float, default=0.1)
    args = parser.parse_args()
    print(args)
    predictor = Predictor(
        threshold=args.thresh, 
        vision_engine=args.vision_engine,
        vision_model_name=args.vision_model_name,
        vision_checkpoint=args.vision_checkpoint
    )

    image = PIL.Image.open(args.image)

    detections = predictor.predict_text(image, texts=args.prompt)

    for detection in detections:
        draw_detection(image, detection)

    image.save("data/basic_usage_out.jpg")