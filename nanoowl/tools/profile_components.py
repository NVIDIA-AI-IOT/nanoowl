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
from nanoowl.utils.module_recorder import ModuleRecorder
from nanoowl.utils.tensorrt import load_image_encoder_engine

owlvit = OwlVit()

image = PIL.Image.open("assets/dogs.jpg")

vision_model_trt = load_image_encoder_engine("data/owlvit_vision_model.engine", owlvit.model.owlvit.vision_model.post_layernorm)

owlvit.model.owlvit.vision_model = vision_model_trt

vision_recorder = ModuleRecorder(owlvit.model.owlvit.vision_model)
text_recorder = ModuleRecorder(owlvit.model.owlvit.text_model)
text_recorder = ModuleRecorder(owlvit.model.owlvit.text_model)
full_recorder = ModuleRecorder(owlvit.model)

# warmup
detections = owlvit.predict(image, texts=["a dog"])
torch.cuda.current_stream().synchronize()

# profile
with vision_recorder, text_recorder, full_recorder:
    detections = owlvit.predict(image, texts=["a dog"])

print(f"VISION MODEL: {vision_recorder.get_elapsed_time()}")
print(f"TEXT MODEL: {text_recorder.get_elapsed_time()}")
print(f"FULL MODEL: {full_recorder.get_elapsed_time()}")
