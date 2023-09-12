import torch
import torch.nn as nn
import argparse
from nanoowl.utils.predictor import Predictor
from nanoowl.models import (
    create_model
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--dynamic_axes", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.parse_args()
    args = parser.parse_args()

    # grab post-layernorm from original owlvit
    predictor = Predictor(vision_engine=None)
    post_layernorm = predictor.model.owlvit.vision_model.post_layernorm

    # wrapper to apply postlayernorm to pooled output (to match owlvit encoder semantics)
    class Wrapper(nn.Module):
        def __init__(self, model, pln):
            super().__init__()
            self.model = model
            self.post_layernorm = pln

        def forward(self, image):
            features, pooled = self.model(image)
            pooled = self.post_layernorm(pooled)
            return features, pooled

    model = create_model(args.model_name)
    model.load_state_dict(torch.load(args.checkpoint)['model'])

    wrapper = Wrapper(model, post_layernorm)
    wrapper = wrapper.cuda().eval()

    data = torch.randn(args.batch_size, 3, 768, 768).cuda()

    output = wrapper(data)

    if args.dynamic_axes:
        dynamic_axes = {
            "image": {0: "batch"},
            "last_hidden_state": {0: "batch"},
            "pooled_output": {0: "batch"},
        }
    else:
        dynamic_axes = {}

    torch.onnx.export(
        wrapper,
        data,
        args.output,
        input_names=["image"],
        output_names=["last_hidden_state", "pooled_output"],
        dynamic_axes=dynamic_axes
    )