import torch
import PIL.Image
import numpy as np
from transformers.models.owlvit.modeling_owlvit import (
    OwlViTForObjectDetection
)
from transformers.models.owlvit.processing_owlvit import (
    OwlViTProcessor
)
from dataclasses import dataclass
from typing import List


__all__ = [
    "OwlPredictor"
]


def _owl_center_to_corners_format_torch(bboxes_center):
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        [
            (center_x - 0.5 * width), 
            (center_y - 0.5 * height), 
            (center_x + 0.5 * width), 
            (center_y + 0.5 * height)
        ],
        dim=-1,
    )
    return bbox_corners


def _owl_get_image_size(hf_name: str):

    image_sizes = {
        "google/owlvit-base-patch32": 768,
        "google/owlvit-base-patch16": 768,
        "google/owlvit-large-patch14": 840,
    }

    return image_sizes[hf_name]


def _owl_get_patch_size(hf_name: str):

    patch_sizes = {
        "google/owlvit-base-patch32": 32,
        "google/owlvit-base-patch16": 16,
        "google/owlvit-large-patch14": 14,
    }

    return patch_sizes[hf_name]


def _owl_normalize_grid_corner_coordinates(num_patches_per_side):

    box_coordinates = np.stack(
        np.meshgrid(np.arange(1, num_patches_per_side + 1), np.arange(1, num_patches_per_side + 1)), axis=-1
    ).astype(np.float32)
    box_coordinates /= np.array([num_patches_per_side, num_patches_per_side], np.float32)

    box_coordinates = box_coordinates.reshape(
        box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
    )
    box_coordinates = torch.from_numpy(box_coordinates)

    return box_coordinates


def _owl_compute_box_bias(num_patches_per_side):
    
    box_coordinates = _owl_normalize_grid_corner_coordinates(num_patches_per_side)
    box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

    box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

    box_size = torch.full_like(box_coord_bias, 1.0 / num_patches_per_side)
    box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

    box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)

    return box_bias


def _owl_box_roi_to_box_global(boxes, rois):
    x0y0 = rois[..., :2]
    x1y1 = rois[..., 2:]
    wh = (x1y1 - x0y0).repeat(1, 1, 2)
    x0y0 = x0y0.repeat(1, 1, 2)
    return (boxes * wh) + x0y0


@dataclass
class OwlEncodeTextOutput:
    text_embeds: torch.Tensor


@dataclass
class OwlEncodeImageOutput:
    image_embeds: torch.Tensor
    image_class_embeds: torch.Tensor
    logit_shift: torch.Tensor
    logit_scale: torch.Tensor
    pred_boxes: torch.Tensor


@dataclass
class OwlDecodeOutput:
    labels: torch.Tensor
    scores: torch.Tensor
    boxes: torch.Tensor
    input_indices: torch.Tensor


class OwlPredictor(torch.nn.Module):
    
    def __init__(self,
            model_name: str = "google/owlvit-base-patch32",
            device: str = "cuda"
        ):

        super().__init__()

        self.image_size = _owl_get_image_size(model_name)
        self.device = device
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.patch_size = _owl_get_patch_size(model_name)
        self.num_patches_per_side = self.image_size // self.patch_size
        self.box_bias = _owl_compute_box_bias(self.num_patches_per_side).to(self.device)
        self.num_patches = (self.num_patches_per_side)**2

    def get_num_patches(self):
        return self.num_patches

    def get_device(self):
        return self.device
        
    def get_image_size(self):
        return (self.image_size, self.image_size)
    
    def encode_text(self, text: List[str]) -> OwlEncodeTextOutput:
        text_input = self.processor(text=text, return_tensors="pt")
        input_ids = text_input['input_ids'].to(self.device)
        attention_mask = text_input['attention_mask'].to(self.device)
        text_outputs = self.model.owlvit.text_model(input_ids, attention_mask)
        text_embeds = text_outputs[1]
        text_embeds = self.model.owlvit.text_projection(text_embeds)
        return OwlEncodeTextOutput(text_embeds=text_embeds)

    def encode_image(self, image: torch.Tensor) -> OwlEncodeImageOutput:
        
        vision_outputs = self.model.owlvit.vision_model(image)
        last_hidden_state = vision_outputs[0]
        image_embeds = self.model.owlvit.vision_model.post_layernorm(last_hidden_state)
        class_token_out = image_embeds[:, :1, :]
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.model.layer_norm(image_embeds)  # 768 dim

        # Box Head
        pred_boxes = self.model.box_head(image_embeds)
        pred_boxes += self.box_bias
        pred_boxes = torch.sigmoid(pred_boxes)
        pred_boxes = _owl_center_to_corners_format_torch(pred_boxes)

        # Class Head
        image_class_embeds = self.model.class_head.dense0(image_embeds)
        logit_shift = self.model.class_head.logit_shift(image_embeds)
        logit_scale = self.model.class_head.logit_scale(image_embeds)
        logit_scale = self.model.class_head.elu(logit_scale) + 1

        output = OwlEncodeImageOutput(
            image_embeds=image_embeds,
            image_class_embeds=image_class_embeds,
            logit_shift=logit_shift,
            logit_scale=logit_scale,
            pred_boxes=pred_boxes
        )

        return output
    
    def decode(self, 
            image_output: OwlEncodeImageOutput, 
            text_output: OwlEncodeTextOutput,
            threshold: float = 0.1
        ) -> OwlDecodeOutput:

        num_input_images = image_output.image_class_embeds.shape[0]

        image_class_embeds = image_output.image_class_embeds
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = text_output.text_embeds
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)
        logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
        logits = (logits + image_output.logit_shift) * image_output.logit_scale
        
        scores_sigmoid = torch.sigmoid(logits)
        scores_max = scores_sigmoid.max(dim=-1)
        labels = scores_max.indices
        scores = scores_max.values

        mask = (scores > threshold)

        input_indices = torch.arange(0, num_input_images, dtype=labels.dtype, device=labels.device)
        input_indices = input_indices[:, None].repeat(1, self.num_patches)

        return OwlDecodeOutput(
            labels=labels[mask],
            scores=scores[mask],
            boxes=image_output.pred_boxes[mask],
            input_indices=input_indices[mask]
        )