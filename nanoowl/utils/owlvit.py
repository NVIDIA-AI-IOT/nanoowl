
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
import numpy as np

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
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", device_map=device)
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", device_map=device)
        self.threshold = threshold
        self.device = device
        self.times = {}
        self.inputs_images = {"pixel_values": torch.randn(1, 3, 768, 768)}

    def predict(self, image: PIL.Image.Image, texts: Sequence[str]):
        torch.cuda.current_stream().synchronize()
        self.times = {}
        self.times['start'] = time.perf_counter_ns()
        inputs_text = self.processor(text=texts, return_tensors="pt")

        torch.cuda.current_stream().synchronize()
        self.times['preprocess_text'] = time.perf_counter_ns()
        # inputs_images = self.processor(images=image, return_tensors="pt")
        # print(inputs_images["pixel_values"].shape)
        # inputs_images  = remap_output(self.inputs_images, "cuda")
        image = np.asarray(image.resize((768,768)))
        image = torch.from_numpy(image).permute(2, 0, 1).cuda()[None, ...].float()
        inputs_images = {"pixel_values": image}#self.inputs_images#{"pixel_values": torch.randn(1, 3, 768, 768)}

        torch.cuda.current_stream().synchronize()
        self.times['preprocess_images'] = time.perf_counter_ns()
        inputs = {}
        inputs.update(inputs_text)
        inputs.update(inputs_images)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        torch.cuda.current_stream().synchronize()
        self.times['move_inputs'] = time.perf_counter_ns()

        outputs = self.model(**inputs)
        
        torch.cuda.current_stream().synchronize()
        self.times['infer'] = time.perf_counter_ns()

        outputs = remap_output(outputs, "cpu")

        torch.cuda.current_stream().synchronize()
        self.times['move_outputs'] = time.perf_counter_ns()

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.threshold)
        torch.cuda.current_stream().synchronize()
        self.times['postprocess'] = time.perf_counter_ns()
        i = 0
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            detection = {"bbox": box.tolist(), "score": float(score), "label": int(label), "text": texts[label]}
            detections.append(detection)
        torch.cuda.current_stream().synchronize()
        self.times['end'] = time.perf_counter_ns()
        return detections