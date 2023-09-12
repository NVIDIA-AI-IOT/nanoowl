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

predictor = Predictor(threshold=0.25, vision_engine="data/owlvit_vision_model.engine")

image = PIL.Image.open("assets/dogs.jpg")


detections = predictor.predict(image, texts="a dog")


def draw_bbox(bbox):
    x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    plt.plot(x, y, 'g-')

plt.imshow(image)
for detection in detections:
    draw_bbox(detection['bbox'])
plt.show()
plt.savefig("data/visualize_owlvit_out.jpg")