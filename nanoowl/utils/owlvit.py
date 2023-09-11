
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

# OwlViTObjectDetectionOutput(
#             image_embeds=feature_map,
#             text_embeds=query_embeds,
#             pred_boxes=pred_boxes,
#             logits=pred_logits,
#             class_embeds=class_embeds,
#             text_model_output=text_outputs,
#             vision_model_output=vision_outputs,
#         )

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

    def predict(self, image: PIL.Image.Image, texts: Sequence[str]):
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        outputs = remap_output(outputs, "cpu")

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.threshold)
        i = 0
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            detection = {"bbox": box.tolist(), "score": float(score), "label": int(label), "text": texts[label]}
            detections.append(detection)
        return detections