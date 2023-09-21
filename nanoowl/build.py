import pytest
import PIL.Image
import os
import argparse
from nanoowl.model import (
    OwlVitPredictor,
    OwlVitImageEncoderModule,
    OwlVitTextEncoderModule
)
from nanoowl.utils.drawing import draw_detections_raw


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="data/owlvit-base-patch32-image-encoder.engine")
    parser.add_argument("--image_encoder_onnx", type=str, default="data/owlvit-base-patch32-image-encoder.onnx")
    # parser.add_argument("--text_encoder_engine", type=str, default="data/owlvit-base-patch32-text-encoder.engine")
    # parser.add_argument("--text_encoder_onnx", type=str, default="data/owlvit-base-patch32-text-encoder.onnx")
    parser.add_argument("--max_num_image", type=int, default=1)
    parser.add_argument("--max_num_text", type=int, default=20)
    parser.add_argument("--skip_image", action="store_true")
    parser.add_argument("--skip_text", action="store_true")
    args = parser.parse_args()

    if args.image_encoder_engine and not args.skip_image:
        OwlVitImageEncoderModule.build_trt(
            args.model,
            args.image_encoder_engine,
            max_batch_size=args.max_num_image,
            onnx_path=args.image_encoder_onnx
        )

    # if args.text_encoder_engine and not args.skip_text:

    #     OwlVitTextEncoderModule.build_trt(
    #         args.model,
    #         args.text_encoder_engine,
    #         max_num_text=args.max_num_text,
    #         onnx_path=args.text_encoder_onnx
    #     )
