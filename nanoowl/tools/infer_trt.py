import torch
import torch.nn as nn
import tensorrt as trt
import PIL.Image
import matplotlib.pyplot as plt
import time
from torch2trt import TRTModule
from nanoowl.utils.predictor import Predictor
from nanoowl.utils.module_recorder import ModuleRecorder
from nanoowl.utils.tensorrt import load_image_encoder_engine

owlvit = Predictor()


image = PIL.Image.open("assets/dogs.jpg")
vision_model_trt = load_image_encoder_engine("data/owlvit_vision_model.engine", owlvit.model.owlvit.vision_model.post_layernorm)

owlvit.model.owlvit.vision_model = vision_model_trt

# warmup
for i in range(3):
    detections = owlvit.predict(image, texts=["a dog"])
torch.cuda.current_stream().synchronize()
count = 15
t0 = time.perf_counter_ns()
for i in range(count):
    detections = owlvit.predict(image, texts=["a dog"])
torch.cuda.current_stream().synchronize()
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9
print(detections[0])
print(count / dt)
