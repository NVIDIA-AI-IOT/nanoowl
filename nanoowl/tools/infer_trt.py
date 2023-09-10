import torch
import torch.nn as nn
import tensorrt as trt
import PIL.Image
import matplotlib.pyplot as plt
from typing import Optional
from transformers.modeling_outputs import BaseModelOutputWithPooling
import time
from torch2trt import TRTModule
from nanoowl.utils.owlvit import OwlVit

owlvit = OwlVit()

def load_image_encoder_engine(path: str, pln):

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(
        engine=engine,
        input_names=["image"],
        output_names=["last_hidden_state", "pooled_output"]
    )

    class Wrapper(nn.Module):
        def __init__(self, trt_module):
            super().__init__()
            self.trt_module = trt_module
            self.post_layernorm = pln

        def forward(self, 
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None):
            in_dev = pixel_values.device
            output = self.trt_module(pixel_values.to("cuda"))
            return BaseModelOutputWithPooling(
                last_hidden_state=output[0].to(in_dev),
                pooler_output=output[1].to(in_dev)
            )


    return Wrapper(image_encoder_trt)

image = PIL.Image.open("assets/dogs.jpg")
vision_model_trt = load_image_encoder_engine("data/owlvit_vision_model.engine", owlvit.model.owlvit.vision_model.post_layernorm)

owlvit.model.owlvit.vision_model = vision_model_trt

count = 5
t0 = time.perf_counter_ns()
for i in range(count):
    detections = owlvit.predict(image, texts=["a dog"])
torch.cuda.current_stream().synchronize()
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9
print(dt / count)
