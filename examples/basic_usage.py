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
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_engine", type=str, default="data/owlvit_vision_model.engine")
    parser.add_argument("--thresh", type=float, default=0.1)
    args = parser.parse_args()

    predictor = Predictor(threshold=args.thresh, vision_engine=args.vision_engine)

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