
import torch
import torch.nn as nn
import tensorrt as trt
from typing import Optional
from torch2trt import TRTModule

from transformers.modeling_outputs import BaseModelOutputWithPooling

def load_image_encoder_engine(path: str, pln=None, apply_pln_to_pooled=False,
        use_wrapper: bool = True):

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(
        engine=engine,
        input_names=["image"],
        output_names=["last_hidden_state", "pooled_output"]
    )

    if use_wrapper:
        class Wrapper(nn.Module):
            def __init__(self, trt_module):
                super().__init__()
                self.trt_module = trt_module
                if pln is not None:
                    self.post_layernorm = pln
                self.apply_pln_to_pooled = apply_pln_to_pooled

            def forward(self, 
                pixel_values: torch.FloatTensor,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):
                in_dev = pixel_values.device
                output = self.trt_module(pixel_values.to("cuda"))

                pooler_output = output[1].to(in_dev)

                if self.apply_pln_to_pooled:
                    pooler_output = self.post_layernorm(pooler_output)

                return BaseModelOutputWithPooling(
                    last_hidden_state=output[0].to(in_dev),
                    pooler_output=pooler_output
                )

        image_encoder_trt = Wrapper(image_encoder_trt)
    return image_encoder_trt