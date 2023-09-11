
import PIL.Image
import torch
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    OwlViTVisionModel
)
from transformers.models.owlvit.modeling_owlvit import (
    OwlViTObjectDetectionOutput
)
from typing import Sequence, List, Tuple
import torch.nn as nn
import time

def remap_output(output, device):
    if isinstance(output, torch.Tensor):
        return output.to(device)
    else:
        res = output.__class__
        resdict = {}
        for k, v in output.items():
            resdict[k] = remap_output(v,device)
        return res(**resdict)

class OwlVit(object):
    def __init__(self, threshold=0.1, device="cuda"):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", device_map="cpu")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", device_map=device)
        self.threshold = threshold
        self.device = device
        self.times = {}

    def predict(self, image: PIL.Image.Image, texts: Sequence[str]):
        self.times = {}
        self.times['start'] = time.perf_counter_ns()
        inputs = self.processor(text=texts, images=image, return_tensors="pt")

        self.times['preprocess'] = time.perf_counter_ns()

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        outputs = remap_output(outputs, "cpu")

        self.times['infer'] = time.perf_counter_ns()

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.threshold)
        self.times['postprocess'] = time.perf_counter_ns()
        i = 0
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            detection = {"bbox": box.tolist(), "score": float(score), "label": int(label), "text": texts[label]}
            detections.append(detection)
        self.times['end'] = time.perf_counter_ns()
        return detections