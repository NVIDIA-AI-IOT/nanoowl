import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from transformers.models.owlvit.modeling_owlvit import (
    OwlViTForObjectDetection
)
from transformers.models.owlvit.processing_owlvit import (
    OwlViTProcessor
)



class OwlVitImageFormatter(object):
    
    def __init__(self, device: str = "cuda"):
        self.device = device

    def __call__(self, image):
        pixel_values = torch.from_numpy(np.asarray(image))[None, ...]
        pixel_values = pixel_values.permute(0, 3, 1, 2)
        pixel_values = pixel_values.to(self.device).float()
        return pixel_values


class OwlVitTextTokenizer(object):
    def __init__(self, hf_preprocessor, device: str = "cuda"):
        self.hf_preprocessor = hf_preprocessor
        self.device = device

    def __call__(self, text):
        text_input = self.hf_preprocessor(text=text, return_tensors="pt")
        input_ids = text_input['input_ids'].to(self.device)
        attention_mask = text_input['attention_mask'].to(self.device)
        return input_ids, attention_mask


class OwlVitImagePreprocessorModule(nn.Module):
    def __init__(self, image_size=768):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None] * 255.
        )
        self.register_buffer(
            "std", torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None] * 255.
        )
        self.image_size = image_size

    def forward(self, image):
        pixel_values = F.interpolate(image, (self.image_size, self.image_size), mode="bilinear")
        pixel_values.sub_(self.mean).div_(self.std)
        return pixel_values
    
# class OwlVitImagePreprocessor(object):

#     def 
#     def export_onnx(self, path: str):

#         model = self.eval().to("cpu")

#         data = torch.randn(1, 3, self.image_size, self.image_size).to("cpu")

#         dynamic_axes = {
#             "image": {0: "batch", 2: "height", 3: "width"},
#             "image_preprocessed": {0: "batch"}
#         }

#         torch.onnx.export(
#             model, 
#             data, 
#             path, 
#             input_names=["image"], 
#             output_names=["image_preprocessed"],
#             dynamic_axes=dynamic_axes
#         )


class OwlVitImageEncoderModule(nn.Module):
    def __init__(self, vision_model, layer_norm):
        super().__init__()
        self.vision_model = vision_model
        self.layer_norm = layer_norm

    def forward(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values)

        vision_outputs = self.vision_model(pixel_values)

        last_hidden_state = vision_outputs[0]
        image_embeds = self.vision_model.post_layernorm(last_hidden_state)

        class_token_out = image_embeds[:, :1, :]

        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        return image_embeds


class OwlVitTextEncoderModule(nn.Module):
    def __init__(self, text_model, text_projection):
        super().__init__()
        self.text_model = text_model
        self.text_projection = text_projection

    def forward(self, input_ids, attention_mask):
        text_outputs = self.text_model(input_ids, attention_mask)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        return text_embeds


class OwlVitBoxPredictorModule(nn.Module):
    def __init__(self, box_head, num_patches):
        super().__init__()
        self.box_head = box_head
        self.register_buffer(
            "box_bias",
            self._compute_box_bias(num_patches)
        )

    def _normalize_grid_corner_coordinates(self, num_patches):

        box_coordinates = np.stack(
            np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1
        ).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.reshape(
            box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
        )
        box_coordinates = torch.from_numpy(box_coordinates)

        return box_coordinates

    def _compute_box_bias(self, num_patches):
        box_coordinates = self._normalize_grid_corner_coordinates(num_patches)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0 / num_patches)
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    def forward(self, image_embeds):
        pred_boxes = self.box_head(image_embeds)
        pred_boxes += self.box_bias
        pred_boxes = torch.sigmoid(pred_boxes)
        return pred_boxes


class OwlVitClassPredictorModule(nn.Module):
    def __init__(self, class_head):
        super().__init__()
        self.class_head = class_head

    def forward(self, image_embeds, query_embeds, query_mask):

        pred_logits, class_embeds = self.class_head(
            image_embeds,
            query_embeds,
            query_mask
        )

        return pred_logits, class_embeds
    

def center_to_corners_format_torch(bboxes_center):
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners


class OwlVitDetectionPostprocessor(object):

    def __call__(self, threshold, logits, boxes, target_sizes):
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format_torch(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            img_h, img_w = target_sizes.unbind(1)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]

        results.append({"scores": score, "labels": label, "boxes": box})

        return results

def get_patch_size(pretrained_name: str):

    patch_sizes = {
        "google/owlvit-base-patch32": 32,
        "google/owlvit-base-patch16": 16,
        "google/owlvit-large-patch14": 14,
    }

    return patch_sizes[pretrained_name]


def get_image_size(pretrained_name: str):

    image_sizes = {
        "google/owlvit-base-patch32": 768,
        "google/owlvit-base-patch16": 768,
        "google/owlvit-large-patch14": 840,
    }

    return image_sizes[pretrained_name]


class OwlVitBuilder(object):

    def __init__(self,
            pretrained_name: str = "google/owlvit-base-patch32",
        ):
        self.pretrained_name = pretrained_name
        self.image_size = get_image_size(pretrained_name)
        self.patch_size = get_patch_size(pretrained_name)

    def build_hf_model(self):
        model = OwlViTForObjectDetection.from_pretrained(
            self.pretrained_name
        )
        return model

    def build_hf_processor(self):
        processor = OwlViTProcessor.from_pretrained(
            self.pretrained_name
        )
        return processor
    
    def build_image_encoder_module(self, hf_model=None):
        if hf_model is None:
            hf_model = self.build_hf_model()
        return OwlVitImageEncoderModule(hf_model.owlvit.vision_model, hf_model.layer_norm)

    def build_text_encoder_module(self, hf_model=None):
        if hf_model is None:
            hf_model = self.build_hf_model()
        return OwlVitTextEncoderModule(hf_model.owlvit.text_model, hf_model.owlvit.text_projection)

    def build_box_predictor_module(self, hf_model=None):
        if hf_model is None:
            hf_model = self.build_hf_model()
        return OwlVitBoxPredictorModule(
            hf_model.box_head,
            num_patches=self.image_size // self.patch_size
        )

    def build_class_predictor_module(self, hf_model=None):
        if hf_model is None:
            hf_model = self.build_hf_model()
        return OwlVitClassPredictorModule(hf_model.class_head)

    def build_image_preprocessor_module(self):
        return OwlVitImagePreprocessorModule(self.image_size)
    
    def build_image_formatter(self, device: str = "cuda"):
        return OwlVitImageFormatter(device)
    
    def build_text_tokenizer(self, hf_processor=None, device: str = "cuda"):
        if hf_processor is None:
            hf_processor = self.build_hf_processor()
        return OwlVitTextTokenizer(hf_processor, device)
    
    def build_torch_predictor(self, device: str = "cuda"):
        hf_model = self.build_hf_model()
        hf_processor = self.build_hf_processor()
        predictor = OwlVitPredictor(
            image_encoder=self.build_image_encoder_module(hf_model).eval().to(device),
            text_encoder=self.build_text_encoder_module(hf_model).eval().to(device),
            box_predictor=self.build_box_predictor_module(hf_model).eval().to(device),
            class_predictor=self.build_class_predictor_module(hf_model).eval().to(device),
            image_preprocessor=self.build_image_preprocessor_module().eval().to(device),
            image_formatter=self.build_image_formatter(device),
            text_tokenizer=self.build_text_tokenizer(hf_processor, device),
            detection_postprocessor=OwlVitDetectionPostprocessor()
        )
        return predictor


class OwlVitPredictor(object):
    def __init__(self,
            image_encoder: OwlVitImageEncoderModule,
            text_encoder: OwlVitTextEncoderModule,
            box_predictor: OwlVitBoxPredictorModule,
            class_predictor: OwlVitClassPredictorModule,
            image_preprocessor: OwlVitImagePreprocessorModule,
            image_formatter: OwlVitImageFormatter,
            text_tokenizer: OwlVitTextTokenizer,
            detection_postprocessor: OwlVitDetectionPostprocessor
        ):

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.box_predictor = box_predictor
        self.class_predictor = class_predictor
        self.image_preprocessor = image_preprocessor
        self.image_formatter = image_formatter
        self.text_tokenizer = text_tokenizer
        self.detection_postprocessor = detection_postprocessor

    @staticmethod
    def from_pretrained(name: str):
        builder = OwlVitBuilder(name)
        return builder.build_torch_predictor()
    
    def predict(self, image=None, text=None, threshold=0.1):
        #TODO: support multi-batch inference

        # Encode Text
        input_ids, attention_mask = self.text_tokenizer(text)
        text_embeds = self.text_encoder(input_ids, attention_mask)
        query_mask = input_ids[None, :, 0] > 0
        query_embeds = text_embeds[None, ...]

        # Encode Image
        image = self.image_formatter(image)
        image_resized = self.image_preprocessor(image)
        image_embeds = self.image_encoder(image_resized)

        # Predict boxes
        pred_boxes = self.box_predictor(image_embeds)

        # Predict classes
        pred_logits, class_embeds = self.class_predictor(image_embeds, query_embeds, query_mask)

        # Decode
        image_sizes = torch.Tensor([[image.size(2), image.size(3)]]).to(image.device)

        detections = self.detection_postprocessor(
            threshold=threshold, 
            logits=pred_logits,
            boxes=pred_boxes,
            target_sizes=image_sizes
        )

        return detections