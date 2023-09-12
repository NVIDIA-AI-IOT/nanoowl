import torch
from nanoowl.utils.predictor import Predictor

predictor = Predictor()

out = predictor.encode_text(texts=["a dog", "a cat"])

input_ids = torch.ones(1, 16).long().cuda()
attention_mask = torch.randn(1, 16).cuda()
pixel_values = torch.randn(1, 3, 768, 768).cuda()

module = predictor.model.owlvit

# module(input_ids, pixel_values, attention_mask)

torch.onnx.export(
    module,
    (input_ids, pixel_values, attention_mask),
    "data/test_out.onnx",
    opset_version=16
)