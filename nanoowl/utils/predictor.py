
from typing import Sequence

import numpy as np
import PIL.Image
import torch
from transformers import (
    OwlViTForObjectDetection, 
    OwlViTProcessor
)
from transformers.models.owlvit.modeling_owlvit import (
    OwlViTObjectDetectionOutput,
    OwlViTImageGuidedObjectDetectionOutput
)
import transformers.models.owlvit.image_processing_owlvit
from nanoowl.models import create_model
from nanoowl.utils.tensorrt import load_image_encoder_engine
from nanoowl.utils.transform import build_owlvit_vision_transform


def remap_device(output, device):
    if isinstance(output, torch.Tensor):
        return output.to(device)
    else:
        res = output.__class__
        resdict = {}
        for k, v in output.items():
            resdict[k] = remap_device(v,device)
        return res(**resdict)


def load_owlvit_model(
        device="cuda", 
        vision_engine=None,
        vision_checkpoint=None,
        vision_model_name=None,
        pretrained_name="google/owlvit-base-patch32"
    ):
    
    model = OwlViTForObjectDetection.from_pretrained(pretrained_name, device_map=device)

    # Overwrite with different vision encoder
    if vision_model_name is not None:
        assert vision_checkpoint is not None
        vision_model = create_model(vision_model_name)
        vision_model.load_state_dict(torch.load(vision_checkpoint)['model'])
        vision_model = vision_model.eval().to(device)
        model.owlvit.vision_model = vision_model

    # Overwrite with engine
    if vision_engine is not None:
        vision_model_trt = load_image_encoder_engine(vision_engine, model.owlvit.vision_model.post_layernorm)
        model.owlvit.vision_model = vision_model_trt
    
    return model


class Predictor(object):
    def __init__(self, 
            threshold=0.1, 
            device="cuda", 
            vision_engine=None,
            vision_checkpoint=None,
            vision_model_name=None,
            query_image_nms_threshold=0.3,
            patch_size: int = 32
        ):
        if patch_size == 16:
            pretrained_name = "google/owlvit-base-patch16"
        elif patch_size == 32:
            pretrained_name = "google/owlvit-base-patch32"

        self._transform = build_owlvit_vision_transform(device)
        self.processor = OwlViTProcessor.from_pretrained(pretrained_name, device_map=device)
        self.model = load_owlvit_model(
            device,
            vision_engine,
            vision_checkpoint,
            vision_model_name,
            pretrained_name=pretrained_name
        )
        self.threshold = threshold
        self.device = device
        self.query_image_nms_threshold = query_image_nms_threshold
        self.patch_size = patch_size
        self.pretrained_name = pretrained_name

        # state
        self._input_ids = None
        self._image = None
        self._pixel_values = None
        self._attention_mask = None
        self._image_embeds = None # flat
        self._feature_map = None # spatial
        self._vision_outputs = None
        self._pred_boxes = None
        self._text_embeds = None
        self._text_outputs = None
        self._text = None
        self._box_bias = self._compute_box_bias(
            num_patches=768//patch_size, 
            device=self.device
        )

    def _normalize_grid_corner_coordinates(self, num_patches, device):

        box_coordinates = np.stack(
            np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1
        ).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.reshape(
            box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
        )
        box_coordinates = torch.from_numpy(box_coordinates).to(device)

        return box_coordinates

    def _compute_box_bias(self, num_patches, device):
        box_coordinates = self._normalize_grid_corner_coordinates(num_patches, device)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0 / num_patches)
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    @torch.no_grad()
    def box_predictor(self, image_feats, feature_map):
        # use cached box bias
        pred_boxes = self.model.box_head(image_feats)
        pred_boxes += self._box_bias
        pred_boxes = torch.sigmoid(pred_boxes)
        return pred_boxes

    @torch.no_grad()
    def embed_image(self, pixel_values):
        
        vision_outputs = self.model.owlvit.vision_model(pixel_values)

        last_hidden_state = vision_outputs[0]
        image_embeds = self.model.owlvit.vision_model.post_layernorm(last_hidden_state)

        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.model.layer_norm(image_embeds)

        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )

        feature_map = image_embeds.reshape(new_size)

        return image_embeds, feature_map, vision_outputs

    @torch.no_grad()
    def embed_text(self, input_ids, attention_mask):
        
        text_outputs = self.model.owlvit.text_model(input_ids, attention_mask)
        text_embeds = text_outputs[1]
        text_embeds = self.model.owlvit.text_projection(text_embeds)

        return text_embeds, text_outputs

    @torch.no_grad()
    def embed_image_text(self, input_ids, pixel_values, attention_mask):
        text_embeds, text_outputs = self.embed_text(input_ids, attention_mask)
        vision_embeds, vision_outputs = self.embed_image(pixel_values)
        return text_embeds, text_outputs, vision_embeds, vision_outputs
    

    @torch.no_grad()
    def set_image(self, image):
        pixel_values = self._transform(image)[None, ...].to(self.device)
        image_embeds, feature_map, vision_outputs = self.embed_image(pixel_values)
        self._pixel_values = pixel_values
        self._image_embeds = image_embeds
        self._feature_map = feature_map
        self._vision_outputs = vision_outputs
        self._image = image
        self._pred_boxes = self.box_predictor(self._image_embeds, self._feature_map)

    @torch.no_grad()
    def set_text(self, text):

        if isinstance(text, str):
            text = [text]

        text_input = self.processor(text=text, return_tensors="pt")
        input_ids = text_input['input_ids'].to(self.device)
        attention_mask = text_input['attention_mask'].to(self.device)
        text_embeds, text_outputs = self.embed_text(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        self._text_embeds = text_embeds
        self._text_outputs = text_outputs
        self._text = text

    @torch.no_grad()
    def set_query_image(self, query_image):
        pixel_values = self._transform(query_image)[None, ...].to(self.device)
        image_embeds, feature_map, vision_outputs = self.embed_image(pixel_values)
        query_embeds, best_box_indices, query_pred_boxes = self.model.embed_image_query(
            image_embeds, feature_map
        )
        self._query_embeds = query_embeds
        self._query_pred_boxes = query_pred_boxes
        self._query_feature_map = feature_map
        self._query_image = query_image
        self._query_pixel_values = pixel_values
        
    @torch.no_grad()
    def _process_query_image(self):
        
        (pred_logits, class_embeds) = self.model.class_predictor(
            self._image_embeds, self._query_embeds
        )

        return OwlViTImageGuidedObjectDetectionOutput(
            image_embeds=self._feature_map,
            query_image_embeds=self._query_feature_map,
            target_pred_boxes=self._pred_boxes,
            query_pred_boxes=self._query_pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds
        )
    
    @torch.no_grad()
    def predict_query_image(self, image: PIL.Image.Image=None, query_image: PIL.Image.Image = None):

        if image is not None:
            self.set_image(image)
        else:
            image = self._image
        
        if query_image is not None:
            self.set_query_image(query_image)

        # Run model
        outputs = self._process_query_image()

        # Copy outputs to CPU (for postprocessing)
        outputs = remap_device(outputs, "cpu")

        # Postprocess output
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_image_guided_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=self.threshold,
            nms_threshold=self.query_image_nms_threshold
        )
        # Format output
        i = 0
        boxes, scores = results[i]["boxes"], results[i]["scores"]
        detections = []
        for box, score in zip(boxes, scores):
            detection = {"bbox": box.tolist(), "score": float(score)}
            detections.append(detection)

        return detections

    @torch.no_grad()
    def _process_text(self):

        input_ids = self._input_ids
        query_embeds = self._text_embeds
        feature_map = self._feature_map
        image_feats = self._image_embeds

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape

        # Reshape from [batch_size * max_text_queries, hidden_dim] -> [batch_size, max_text_queries, hidden_dim]
        max_text_queries = input_ids.shape[0] // batch_size
        query_embeds = query_embeds.reshape(batch_size, max_text_queries, query_embeds.shape[-1])

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
        query_mask = input_ids[..., 0] > 0

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.model.class_predictor(image_feats, query_embeds, query_mask)

        # Predict object boxes
        pred_boxes = self._pred_boxes

        return OwlViTObjectDetectionOutput(
            image_embeds=feature_map,
            text_embeds=query_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds
        )

    @torch.no_grad()
    def predict_text(self, image: PIL.Image.Image=None, text: Sequence[str]=None):

        if isinstance(text, str):
            text = [text]

        if image is not None:
            self.set_image(image)
        else:
            image = self._image
        
        if text is not None:
            self.set_text(text)
        else:
            text = self._text

        # Run model
        outputs = self._process_text()

        # Copy outputs to CPU (for postprocessing)
        outputs = remap_device(outputs, "cpu")

        # Postprocess output
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.threshold)

        # Format output
        i = 0
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            detection = {"bbox": box.tolist(), "score": float(score), "label": int(label), "text": text[label]}
            detections.append(detection)

        return detections