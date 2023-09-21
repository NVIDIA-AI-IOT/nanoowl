import argparse
import PIL.Image
from nanoowl.model import (
    OwlVitPredictor
)
from nanoowl.utils.drawing import draw_detections_raw


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="assets/owl_glove.jpg")
    parser.add_argument("--text", action='append', required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="data/owl_glove_out.jpg")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="data/owlvit-base-patch32-image-encoder.engine")
    parser.add_argument("--text_encoder_engine", type=str, default="data/owlvit-base-patch32-text-encoder.engine")
    parser.add_argument("--output_path", type=str, default="data/predict_out.jpg")
    args = parser.parse_args()

    if args.image_encoder_engine.lower() == "none":
        args.image_encoder_engine = None

    if args.text_encoder_engine.lower() == "none":
        args.text_encoder_engine = None
        
    predictor = OwlVitPredictor.from_pretrained(
        args.model,
        image_encoder_engine=args.image_encoder_engine,
        text_encoder_engine=args.text_encoder_engine,
        device="cuda"
    )

    image = PIL.Image.open(args.image)

    text = args.text

    detections = predictor.predict(image=image, text=text, threshold=args.threshold)

    draw_detections_raw(image, detections)

    image.save(args.output)