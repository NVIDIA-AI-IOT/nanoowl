import torch
import argparse
from nanoowl.utils.predictor import Predictor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/owlvit_vision_model.onnx")
    parser.parse_args()
    args = parser.parse_args()

    predictor = Predictor(vision_engine=None)

    data = torch.randn(1, 3, 768, 768).cuda()

    vision_model = predictor.model.owlvit.vision_model.cuda().eval()

    output = vision_model(data)

    torch.onnx.export(
        vision_model,
        data,
        args.output,
        input_names=["image"],
        output_names=["last_hidden_state", "pooled_output"]
    )