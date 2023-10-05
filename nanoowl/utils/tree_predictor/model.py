import torch
from typing import List, Tuple, Union, Optional
import numpy as np
import torch
import clip
import PIL.Image
import PIL.ImageDraw
from enum import Enum
from torchvision.ops import roi_align
from transformers.models.owlvit.modeling_owlvit import (
    OwlViTForObjectDetection
)
from transformers.models.owlvit.processing_owlvit import (
    OwlViTProcessor
)
from dataclasses import dataclass, field


DEFAULT_CLIP_IMAGE_MEAN = [0.48145466 * 255., 0.4578275 * 255., 0.40821073 * 255.]
DEFAULT_CLIP_IMAGE_STD = [0.26862954 * 255., 0.26130258 * 255., 0.27577711 * 255.]


def draw_box(image, box, color="green"):
    draw = PIL.ImageDraw.Draw(image)
    draw.rectangle(box, outline=color)
    return image


def center_to_corners_format_torch(bboxes_center):
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


def owl_get_image_size(hf_name: str):

    image_sizes = {
        "google/owlvit-base-patch32": 768,
        "google/owlvit-base-patch16": 768,
        "google/owlvit-large-patch14": 840,
    }

    return image_sizes[hf_name]


def owl_get_patch_size(hf_name: str):

    patch_sizes = {
        "google/owlvit-base-patch32": 32,
        "google/owlvit-base-patch16": 16,
        "google/owlvit-large-patch14": 14,
    }

    return patch_sizes[hf_name]


def owl_normalize_grid_corner_coordinates(num_patches):

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


def owl_compute_box_bias(num_patches):
    box_coordinates = owl_normalize_grid_corner_coordinates(num_patches)
    box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

    # Unnormalize xy
    box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

    # The box size is biased to the patch size
    box_size = torch.full_like(box_coord_bias, 1.0 / num_patches)
    box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

    # Compute box bias
    box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
    return box_bias


def xyxy_to_xywh(box):
    x0 = box[..., 0]
    y0 = box[..., 1]
    x1 = box[..., 2]
    y1 = box[..., 3]
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = (x1 - x0)
    h = (y1 - y0)
    box_new = torch.stack([cx, cy, w, h], dim=-1)
    return box_new

def box_roi_to_box_global(boxes, rois):
    x0y0 = rois[..., :2]
    x1y1 = rois[..., 2:]
    wh = (x1y1 - x0y0).repeat(1, 1, 2)
    x0y0 = x0y0.repeat(1, 1, 2)
    return (boxes * wh) + x0y0


@dataclass
class ClipDecodeOutput:
    labels: torch.Tensor
    scores: torch.Tensor


@dataclass
class OwlImageEncodeOutput:
    image_embeds: torch.Tensor
    image_class_embeds: torch.Tensor
    logit_shift: torch.Tensor
    logit_scale: torch.Tensor
    pred_boxes: torch.Tensor

    def __getitem__(self, idx):
        return OwlImageEncodeOutput(
            image_embeds=self.image_embeds[idx],
            image_class_embeds=self.image_class_embeds[idx],
            logit_shift=self.logit_shift[idx],
            logit_scale=self.logit_scale[idx],
            pred_boxes=self.pred_boxes[idx]
        )

    def concat(self, other: "OwlImageEncodeOutput"):
        return OwlImageEncodeOutput(
            image_embeds=torch.cat([self.image_embeds, other.image_embeds], dim=0),
            image_class_embeds=torch.cat([self.image_class_embeds, other.image_class_embeds], dim=0),
            logit_shift=torch.cat([self.logit_shift, other.logit_shift], dim=0),
            logit_scale=torch.cat([self.logit_scale, other.logit_scale], dim=0),
            pred_boxes=torch.cat([self.pred_boxes, other.pred_boxes], dim=0)
        )

@dataclass
class OwlDecodeOutput:
    labels: torch.Tensor
    scores: torch.Tensor
    boxes: torch.Tensor
    roi_indices: torch.Tensor


class TreeModel(torch.nn.Module):

    def __init__(self,
            clip_model: str = "ViT-B/32",
            clip_size: Tuple[int, int] = (224, 224),
            device: str = "cuda",
            image_mean = DEFAULT_CLIP_IMAGE_MEAN,
            image_std = DEFAULT_CLIP_IMAGE_STD,
            owl_model: str = "google/owlvit-base-patch32",
            owl_image_encoder_engine: Optional[str] = None,
            owl_image_encoder_engine_max_batch_size: int = 1,
            clip_image_encoder_engine: Optional[str] = None,
            clip_image_encoder_engine_max_batch_size: int = 1
        ):
        super().__init__()

        # Common
        self._device = device
        self._image_mean = torch.tensor(image_mean)[None, :, None, None].to(device)
        self._image_std = torch.tensor(image_std)[None, :, None, None].to(device)

        # CLIP
        self._clip_model, _ = clip.load(clip_model, device=device)
        self._clip_size = clip_size

        # OWL

        if owl_image_encoder_engine is not None:
            self._owl_image_encoder_engine = self.load_owl_image_encoder_engine(
                owl_image_encoder_engine,
                owl_image_encoder_engine_max_batch_size
            )
        else:
            self._owl_image_encoder_engine = None
        
        if clip_image_encoder_engine is not None:
            self._clip_image_encoder_engine = self.load_clip_image_encoder_engine(
                clip_image_encoder_engine,
                clip_image_encoder_engine_max_batch_size
            )
        else:
            self._clip_image_encoder_engine = None

        self._owl_model = OwlViTForObjectDetection.from_pretrained(
            owl_model
        ).to(self._device).eval()
        self._owl_processor = OwlViTProcessor.from_pretrained(
            owl_model
        )
        self._owl_size = owl_get_image_size(owl_model)
        self._owl_patch_size = owl_get_patch_size(owl_model)
        self._owl_num_patches = self._owl_size // self._owl_patch_size
        self._owl_box_bias = owl_compute_box_bias(self._owl_num_patches).to(self._device)
        self._owl_num_pred_boxes = (self._owl_num_patches)**2

        # mesh grid used for computing ROI masks
        self._clip_mesh_grid = torch.stack(
            torch.meshgrid(
                torch.linspace(0., 1., self._clip_size[1]),
                torch.linspace(0., 1., self._clip_size[0])
            )
        ).to(self._device).float()
        self._owl_mesh_grid = torch.stack(
            torch.meshgrid(
                torch.linspace(0., 1., self._owl_size),
                torch.linspace(0., 1., self._owl_size)
            )
        ).to(self._device).float()

    def export_clip_image_encoder_onnx(self, path: str):
        raise NotImplementedError

    @staticmethod
    def owl_image_encoder_input_names():
        return ["image"]

    @staticmethod
    def owl_image_encoder_output_names():
        names = [
            "image_embeds",
            "image_class_embeds",
            "logit_shift",
            "logit_scale",
            "pred_boxes"
        ]
        return names


    def export_owl_image_encoder_onnx(self, 
            output_path: str,
            use_dynamic_axes: bool = True,
            batch_size: int = 1,
            onnx_opset=17
        ):
        
        class TempModule(torch.nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
            def forward(self, image):
                output = self.parent.owl_encode_image_torch(image)
                return (
                    output.image_embeds,
                    output.image_class_embeds,
                    output.logit_shift,
                    output.logit_scale,
                    output.pred_boxes
                )

        data = torch.randn(batch_size, 3, self._owl_size, self._owl_size).to(self._device)

        if use_dynamic_axes:
            dynamic_axes =  {
                "image": {0: "batch"},
                "image_embeds": {0: "batch"},
                "image_class_embeds": {0: "batch"},
                "logit_shift": {0: "batch"},
                "logit_scale": {0: "batch"},
                "pred_boxes": {0: "batch"}       
            }
        else:
            dynamic_axes = {}

        model = TempModule(self)

        torch.onnx.export(
            model, 
            data, 
            output_path, 
            input_names=self.owl_image_encoder_input_names(), 
            output_names=self.owl_image_encoder_output_names(),
            dynamic_axes=dynamic_axes,
            opset_version=17
        )
    
    @staticmethod
    def load_owl_image_encoder_engine(engine_path: str, max_batch_size: int):
        import tensorrt as trt
        from torch2trt import TRTModule

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        base_module = TRTModule(
            engine,
            input_names=TreeModel.owl_image_encoder_input_names(),
            output_names=TreeModel.owl_image_encoder_output_names()
        )

        class Wrapper(torch.nn.Module):
            def __init__(self, base_module: TRTModule, max_batch_size: int):
                super().__init__()
                self.base_module = base_module
                self.max_batch_size = max_batch_size

            @torch.no_grad()
            def forward(self, image):

                b = image.shape[0]

                results = []

                for start_index in range(0, b, self.max_batch_size):
                    end_index = min(b, start_index + self.max_batch_size)
                    image_slice = image[start_index:end_index]
                    output = self.base_module(image_slice)
                    results.append(
                        output
                    )

                return OwlImageEncodeOutput(
                    image_embeds=torch.cat([r[0] for r in results], dim=0),
                    image_class_embeds=torch.cat([r[1] for r in results], dim=0),
                    logit_shift=torch.cat([r[2] for r in results], dim=0),
                    logit_scale=torch.cat([r[3] for r in results], dim=0),
                    pred_boxes=torch.cat([r[4] for r in results], dim=0)
                )

        image_encoder = Wrapper(base_module, max_batch_size)

        return image_encoder


    @staticmethod
    def clip_image_encoder_input_names():
        return ["image"]

    @staticmethod
    def clip_image_encoder_output_names():
        return ["image_embeds"]

    def export_clip_image_encoder_onnx(self, 
            output_path: str,
            use_dynamic_axes: bool = True,
            batch_size: int = 1,
            onnx_opset=17
        ):
        
        class TempModule(torch.nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
            def forward(self, image):
                output = self.parent.clip_encode_image(image)
                return output

        data = torch.randn(batch_size, 3, self._clip_size[0], self._clip_size[1]).to(self._device)

        if use_dynamic_axes:
            dynamic_axes =  {
                "image": {0: "batch"},
                "image_embeds": {0: "batch"}
            }
        else:
            dynamic_axes = {}

        model = TempModule(self)

        torch.onnx.export(
            model,
            data,
            output_path,
            input_names=self.clip_image_encoder_input_names(), 
            output_names=self.clip_image_encoder_output_names(),
            dynamic_axes=dynamic_axes,
            opset_version=18
        )

    @staticmethod
    def load_clip_image_encoder_engine(engine_path: str, max_batch_size: int):
        import tensorrt as trt
        from torch2trt import TRTModule

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        base_module = TRTModule(
            engine,
            input_names=TreeModel.clip_image_encoder_input_names(),
            output_names=TreeModel.clip_image_encoder_output_names()
        )

        class Wrapper(torch.nn.Module):
            def __init__(self, base_module: TRTModule, max_batch_size: int):
                super().__init__()
                self.base_module = base_module
                self.max_batch_size = max_batch_size

            @torch.no_grad()
            def forward(self, image):

                b = image.shape[0]

                results = []

                for start_index in range(0, b, self.max_batch_size):
                    end_index = min(b, start_index + self.max_batch_size)
                    image_slice = image[start_index:end_index]
                    output = self.base_module(image_slice)
                    results.append(
                        output
                    )

                return torch.cat(results, dim=0)

        image_encoder = Wrapper(base_module, max_batch_size)

        return image_encoder


    @torch.no_grad()
    def preprocess_image(self, image: Union[PIL.Image.Image, np.ndarray]):
        image = torch.from_numpy(np.asarray(image))
        image = image.permute(2, 0, 1).to(self._device).float()[None, ...]
        return image.sub_(self._image_mean).div_(self._image_std)

    @torch.no_grad()
    def clip_encode_text(self, text: List[str]):
        text_tokens = clip.tokenize(text).to(self._device)
        text_embeds = self._clip_model.encode_text(text_tokens)
        return text_embeds

    @torch.no_grad()
    def clip_encode_image_torch(self, image: torch.Tensor):
        image_embeds = self._clip_model.encode_image(image)
        return image_embeds

    @torch.no_grad()
    def clip_encode_image_trt(self, image: torch.Tensor):
        return self._clip_image_encoder_engine(image)

    @torch.no_grad()
    def clip_encode_image(self, image: torch.Tensor):
        if self._clip_image_encoder_engine is not None:
            return self.clip_encode_image_trt(image)
        else:
            return self.clip_encode_image_torch(image)

    @torch.no_grad()
    def clip_extract_rois(self, image: torch.Tensor, rois: torch.Tensor, pad_square: bool = True, padding_scale: float=1.2):
        if len(rois) == 0:
            return torch.empty(
                (0, image.shape[1], self._clip_size[0], self._clip_size[1]),
                dtype=image.dtype,
                device=image.device
            )

        if pad_square:
            # pad square
            w = padding_scale*(rois[..., 2] - rois[..., 0]) / 2
            h = padding_scale*(rois[..., 3] - rois[..., 1]) / 2
            cx = (rois[..., 0] + rois[..., 2]) / 2
            cy = (rois[..., 1] + rois[..., 3]) / 2
            s = torch.max(w, h)
            rois = torch.stack([cx-s, cy-s, cx+s, cy+s], dim=-1)

            # compute mask
            pad_x = (s - w) / (2 * s)
            pad_y = (s - h) / (2 * s)
            mask_x = (self._clip_mesh_grid[1][None, ...] > pad_x[..., None, None]) & (self._clip_mesh_grid[1][None, ...] < (1. - pad_x[..., None, None]))
            mask_y = (self._clip_mesh_grid[0][None, ...] > pad_y[..., None, None]) & (self._clip_mesh_grid[0][None, ...] < (1. - pad_y[..., None, None]))
            mask = (mask_x & mask_y)

        roi_images = roi_align(image, [rois], output_size=self._clip_size)

        if pad_square:
            roi_images = roi_images * mask[:, None, :, :]

        return roi_images, rois
    
    @torch.no_grad()
    def clip_encode_rois(self, image: torch.Tensor, rois: torch.Tensor):
        roi_images, rois = self.clip_extract_rois(image, rois)
        return self.clip_encode_image(roi_images)
    
    @torch.no_grad()
    def clip_decode(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor):
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
        logit_scale = self._clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        probs = torch.softmax(logits_per_image, dim=-1)
        prob_max = probs.max(dim=-1)
        
        return ClipDecodeOutput(
            labels=prob_max.indices,
            scores=prob_max.values
        )

    @torch.no_grad()
    def owl_encode_text(self, text: List[str]):
        text_input = self._owl_processor(text=text, return_tensors="pt")
        input_ids = text_input['input_ids'].to(self._device)
        attention_mask = text_input['attention_mask'].to(self._device)
        text_outputs = self._owl_model.owlvit.text_model(input_ids, attention_mask)
        text_embeds = text_outputs[1]
        text_embeds = self._owl_model.owlvit.text_projection(text_embeds)
        return text_embeds

    @torch.no_grad()
    def owl_encode_image_torch(self, image: torch.Tensor) -> OwlImageEncodeOutput:

        vision_outputs = self._owl_model.owlvit.vision_model(image)
        last_hidden_state = vision_outputs[0]
        image_embeds = self._owl_model.owlvit.vision_model.post_layernorm(last_hidden_state)
        class_token_out = image_embeds[:, :1, :]
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self._owl_model.layer_norm(image_embeds)  # 768 dim

        # Box Head
        pred_boxes = self._owl_model.box_head(image_embeds)
        pred_boxes += self._owl_box_bias
        pred_boxes = torch.sigmoid(pred_boxes)
        pred_boxes = center_to_corners_format_torch(pred_boxes)

        # Class Head
        image_class_embeds = self._owl_model.class_head.dense0(image_embeds)
        logit_shift = self._owl_model.class_head.logit_shift(image_embeds)
        logit_scale = self._owl_model.class_head.logit_scale(image_embeds)
        logit_scale = self._owl_model.class_head.elu(logit_scale) + 1

        output = OwlImageEncodeOutput(
            image_embeds=image_embeds,
            image_class_embeds=image_class_embeds,
            logit_shift=logit_shift,
            logit_scale=logit_scale,
            pred_boxes=pred_boxes
        )

        return output

    @torch.no_grad()
    def owl_encode_image_trt(self, image: torch.Tensor) -> OwlImageEncodeOutput:
        return self._owl_image_encoder_engine(image)

    @torch.no_grad()
    def owl_encode_image(self, image: torch.Tensor) -> OwlImageEncodeOutput:
        if self._owl_image_encoder_engine is not None:
            return self.owl_encode_image_trt(image)
        else:
            return self.owl_encode_image_torch(image)

    @torch.no_grad()
    def owl_extract_rois(self, image: torch.Tensor, rois: torch.Tensor, pad_square: bool = True, padding_scale:float=1.2):
        if len(rois) == 0:
            return torch.empty(
                (0, image.shape[1], self._owl_size, self._owl_size),
                dtype=image.dtype,
                device=image.device
            )
        if pad_square:
            # pad square
            w = padding_scale * (rois[..., 2] - rois[..., 0]) / 2
            h = padding_scale * (rois[..., 3] - rois[..., 1]) / 2
            cx = (rois[..., 0] + rois[..., 2]) / 2
            cy = (rois[..., 1] + rois[..., 3]) / 2
            s = torch.max(w, h)
            rois = torch.stack([cx-s, cy-s, cx+s, cy+s], dim=-1)

            # compute mask
            pad_x = (s - w) / (2 * s)
            pad_y = (s - h) / (2 * s)
            mask_x = (self._owl_mesh_grid[1][None, ...] > pad_x[..., None, None]) & (self._owl_mesh_grid[1][None, ...] < (1. - pad_x[..., None, None]))
            mask_y = (self._owl_mesh_grid[0][None, ...] > pad_y[..., None, None]) & (self._owl_mesh_grid[0][None, ...] < (1. - pad_y[..., None, None]))
            mask = (mask_x & mask_y)

        # extract rois
        roi_images = roi_align(image, [rois], output_size=(self._owl_size, self._owl_size))

        # mask rois
        if pad_square:
            roi_images = (roi_images * mask[:, None, :, :])

        return roi_images, rois
    
    @torch.no_grad()
    def owl_encode_rois(self, image: torch.Tensor, rois: torch.Tensor):
        roi_images, rois = self.owl_extract_rois(image, rois)
        output = self.owl_encode_image(roi_images)
        pred_boxes = box_roi_to_box_global(output.pred_boxes, rois[:, None, :])
        output.pred_boxes = pred_boxes
        return output
    
    @torch.no_grad()
    def owl_decode(self, 
            image_outputs: OwlImageEncodeOutput,
            roi_query_embeds: Union[torch.Tensor, List[torch.Tensor]],
            threshold: Union[float, torch.Tensor] = 0.1
        ):

        if isinstance(roi_query_embeds, torch.Tensor):
            roi_query_embeds = [roi_query_embeds]
        
        if isinstance(threshold, float):
            threshold = torch.full(
                (len(roi_query_embeds),), 
                threshold, 
                dtype=image_outputs.logit_shift.dtype, 
                device=image_outputs.logit_scale.device
            )

        labels_all = []  # M
        scores_all = []  # M
        boxes_all = []   # M
        roi_index_all = [] # N roi

        num_roi = len(roi_query_embeds)
        assert num_roi == image_outputs.pred_boxes.shape[0]
        assert threshold.shape[0] == num_roi

        for roi_index in range(num_roi):

            # Get slice for ROI
            image_class_embeds = image_outputs.image_class_embeds[roi_index]

            # Compute logits against query embeds for ROI
            image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
            query_embeds = roi_query_embeds[roi_index]
            query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)
            logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
            logits = (logits + image_outputs.logit_shift[roi_index]) * image_outputs.logit_scale[roi_index]

            # Filter logits by threshold
            scores_sigmoid = torch.sigmoid(logits)
            scores_max = scores_sigmoid.max(dim=-1)
            labels = scores_max.indices
            scores = scores_max.values
            valid = scores > threshold[roi_index]


            boxes_all.append(image_outputs.pred_boxes[roi_index][valid])
            lv = labels[valid]
            labels_all.append(lv)
            scores_all.append(scores[valid])
            roi_index_all.append(torch.full_like(lv, roi_index))
            
        boxes_all = torch.concat(boxes_all, dim=0)
        labels_all = torch.concat(labels_all, dim=0)
        scores_all = torch.concat(scores_all, dim=0)
        roi_index_all = torch.concat(roi_index_all, dim=0)
            
        output = OwlDecodeOutput(
            labels=labels_all,
            scores=scores_all,
            boxes=boxes_all,
            roi_indices=roi_index_all
        )

        return output

    def get_example_data(self):

        image = PIL.Image.open("images/class.jpg")
        person_box = (420, 180, 1030, 1000)
        face_box = (620, 200, 830, 500)
        eye_box = [681.0542, 326.5865, 732.2399, 349.9627]

        boxes = [
            person_box,
            face_box,
            eye_box
        ]

        boxes_tensor = torch.tensor(boxes).to(self._device).float()
        image_tensor = self.preprocess_image(image)

        return image, boxes, image_tensor, boxes_tensor

