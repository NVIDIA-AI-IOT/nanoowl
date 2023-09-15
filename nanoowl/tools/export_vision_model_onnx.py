import torch
import argparse
from nanoowl.utils.predictor import Predictor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/owlvit_vision_model.onnx")
    parser.add_argument("--dynamic_axes", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.parse_args()
    args = parser.parse_args()

    predictor = Predictor(vision_engine=None, patch_size=args.patch_size, device="cpu")

    data = torch.randn(args.batch_size, 3, 768, 768)#.cuda()

    vision_model = predictor.model.owlvit.vision_model.eval()

    output = vision_model(data)

    if args.dynamic_axes:
        dynamic_axes = {
            "image": {0: "batch"},
            "last_hidden_state": {0: "batch"},
            "pooled_output": {0: "batch"},
        }
    else:
        dynamic_axes = {}

    torch.onnx.export(
        vision_model,
        data,
        args.output,
        input_names=["image"],
        output_names=["last_hidden_state", "pooled_output"],
        dynamic_axes=dynamic_axes
    )