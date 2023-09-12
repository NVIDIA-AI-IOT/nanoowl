import torch
import torch.nn as nn
import tensorrt as trt
import PIL.Image
import matplotlib.pyplot as plt
from typing import Optional
from transformers.modeling_outputs import BaseModelOutputWithPooling
import time
from torch2trt import TRTModule
from nanoowl.utils.predictor import Predictor
from nanoowl.utils.module_recorder import ModuleRecorder
from nanoowl.utils.tensorrt import load_image_encoder_engine

owlvit = Predictor(vision_engine="data/owlvit_vision_model.engine")

image = PIL.Image.open("assets/dogs.jpg")

# vision_model_trt = load_image_encoder_engine("data/owlvit_vision_model.engine", owlvit.model.owlvit.vision_model.post_layernorm)

# owlvit.model.owlvit.vision_model = vision_model_trt

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

def to_ms(x):
    return x / 1e9
detections = owlvit.predict(image, texts=["a dog"])

print(to_ms(owlvit.times['preprocess_text'] - owlvit.times['start']))
print(to_ms(owlvit.times['preprocess_images'] - owlvit.times['preprocess_text']))
print(to_ms(owlvit.times['move_inputs'] - owlvit.times['preprocess_images']))
print(to_ms(owlvit.times['infer'] - owlvit.times['move_inputs']))
print(to_ms(owlvit.times['move_outputs'] - owlvit.times['infer']))
print(to_ms(owlvit.times['postprocess'] - owlvit.times['move_outputs']))
print(to_ms(owlvit.times['end'] - owlvit.times['postprocess']))

print(detections)