import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import statistics
import functools
import subprocess
import tensorrt as trt
import tempfile
from torch2trt import TRTModule
from dataclasses import dataclass
from transformers.models.owlvit.modeling_owlvit import (
    OwlViTForObjectDetection
)
from transformers.models.owlvit.processing_owlvit import (
    OwlViTProcessor
)
import time
from collections import OrderedDict

HF_DEFAULT = "google/owlvit-base-patch32"


def get_trtexec_executable():
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec):
        trtexec = "trtexec"
    return trtexec


def hf_get_patch_size(hf_name: str):

    patch_sizes = {
        "google/owlvit-base-patch32": 32,
        "google/owlvit-base-patch16": 16,
        "google/owlvit-large-patch14": 14,
    }

    return patch_sizes[hf_name]


def hf_get_image_size(hf_name: str):

    image_sizes = {
        "google/owlvit-base-patch32": 768,
        "google/owlvit-base-patch16": 768,
        "google/owlvit-large-patch14": 840,
    }

    return image_sizes[hf_name]


def hf_get_embed_size(hf_name: str):

    embed_sizes = {
        "google/owlvit-base-patch32": 512,
        "google/owlvit-base-patch16": 512,
        "google/owlvit-large-patch14": 768, #TODO: check patch14 size
    }

    return embed_sizes[hf_name]


def hf_build_model(hf_name):
    model = OwlViTForObjectDetection.from_pretrained(
        hf_name
    )
    return model


def hf_build_processor(hf_name):
    processor = OwlViTProcessor.from_pretrained(
        hf_name
    )
    return processor


def center_to_corners_format_torch(bboxes_center):
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners


class Profiler:

    active_profilers = set()

    def __init__(self):
        self.stack = []
        self.elapsed_times = OrderedDict()

    def __enter__(self, *args, **kwargs):
        Profiler.active_profilers.add(self)
        return self

    def __exit__(self, *args, **kwargs):
        Profiler.active_profilers.remove(self)

    def current_namespace(self):
        return ".".join(self.stack)

    def add_elapsed_time(self, timing_ms):
        namespace = self.current_namespace()
        if namespace not in self.elapsed_times:
            self.elapsed_times[namespace] = [timing_ms]
        else:
            self.elapsed_times[namespace].append(timing_ms)

    def mean_elapsed_times(self):
        times = OrderedDict()
        for k, v in self.elapsed_times.items():
            times[k] = statistics.mean(v)
        return times

    def median_elapsed_times(self):
        times = OrderedDict()
        for k, v in self.elapsed_times.items():
            times[k] = statistics.median(v)
        return times

    def print_mean_elapsed_times_ms(self):
        for k, v in self.mean_elapsed_times().items():
            print(f"{k}: {round(v, 3) / 1e6}")

    def print_median_elapsed_times_ms(self):
        for k, v in self.median_elapsed_times().items():
            print(f"{k}: {round(v, 3) / 1e6}")

    def clear(self):
        self.elapsed_times = OrderedDict()


class Timer:
    
    def __init__(self, scope: str):
        self.scope = scope
        self._t0 = None
        self._t1 = None

    def is_active(self):
        return len(Profiler.active_profilers) > 0
    
    def __enter__(self, *args, **kwargs):
        if not self.is_active():
            return self
        for profiler in Profiler.active_profilers:
            profiler.stack.append(self.scope)
        torch.cuda.current_stream().synchronize()
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *args, **kwargs):
        if not self.is_active():
            return self
        torch.cuda.current_stream().synchronize()
        self._t1 = time.perf_counter_ns()
        elapsed_time = self.get_elapsed_time_ns()
        for profiler in Profiler.active_profilers:
            profiler.add_elapsed_time(elapsed_time)
            profiler.stack.pop()
        return self

    def get_elapsed_time_ns(self):
        return (self._t1 - self._t0)


    def __call__(self, fn):

        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            with self:
                output = fn(*args, **kwargs)
            return output
        
        return _wrapper


def use_timer(fn):
    return Timer(fn.__qualname__)(fn)


def capture_timings():
    return Profiler()


class OwlVitImageFormatter(object):
    
    def __init__(self, device: str = "cuda"):
        self.device = device

    @use_timer
    def __call__(self, image):
        pixel_values = torch.from_numpy(np.asarray(image))[None, ...]
        pixel_values = pixel_values.permute(0, 3, 1, 2)
        pixel_values = pixel_values.to(self.device).float()
        return pixel_values
    

class OwlVitTextTokenizer(object):
    def __init__(self, hf_preprocessor, device: str = "cuda"):
        self.hf_preprocessor = hf_preprocessor
        self.device = device

    @use_timer
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

    @use_timer
    def forward(self, image):
        pixel_values = F.interpolate(image, (self.image_size, self.image_size), mode="bilinear")
        pixel_values.sub_(self.mean).div_(self.std)
        return pixel_values
    
    def export_onnx(self, path: str):

        model = self.eval().to("cpu")

        data = torch.randn(1, 3, self.image_size, self.image_size).to("cpu")

        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "image_preprocessed": {0: "batch"}
        }

        torch.onnx.export(
            model, 
            data, 
            path, 
            input_names=["image"], 
            output_names=["image_preprocessed"],
            dynamic_axes=dynamic_axes
        )


class OwlVitImageEncoderModule(nn.Module):
    def __init__(self, vision_model, layer_norm, image_size):
        super().__init__()
        self.vision_model = vision_model
        self.layer_norm = layer_norm
        self.image_size = image_size

    @use_timer
    def forward(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values)

        vision_outputs = self.vision_model(pixel_values)

        last_hidden_state = vision_outputs[0]
        image_embeds = self.vision_model.post_layernorm(last_hidden_state)

        class_token_out = image_embeds[:, :1, :]

        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        return image_embeds
    
    @staticmethod
    def from_pretrained(hf_name: str, device: str = "cuda"):
        hf_model = hf_build_model(hf_name)
        image_size = hf_get_image_size(hf_name)
        image_encoder = OwlVitImageEncoderModule(
            hf_model.owlvit.vision_model, hf_model.layer_norm, image_size
        ).eval().to(device)
        return image_encoder

    @staticmethod
    def export_onnx(hf_name: str, output_path: str):
        
        model = OwlVitImageEncoderModule.from_pretrained(hf_name, "cpu")
        image_size = hf_get_image_size(hf_name)

        data = torch.randn(1, 3, image_size, image_size).to("cpu")

        dynamic_axes = {
            "image_preprocessed": {0: "batch"},
            "image_embeds": {0: "batch"}
        }

        torch.onnx.export(
            model, 
            data, 
            output_path, 
            input_names=["image_preprocessed"], 
            output_names=["image_embeds"],
            dynamic_axes=dynamic_axes,
            opset_version=17
        )

    @staticmethod
    def build_trt(
            hf_name: str,
            output_path: str,
            min_batch_size: int = 1,
            opt_batch_size: int = 1,
            max_batch_size: int = 1,
            onnx_path: str = None,
            fp16_mode: bool = True
        ):

        image_size = hf_get_image_size(hf_name)

        if onnx_path is None:
            onnx_path = tempfile.mktemp()
        
        if not os.path.exists(onnx_path):
            OwlVitImageEncoderModule.export_onnx(hf_name, onnx_path)

        trtexec = get_trtexec_executable()
        
        args = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={output_path}",
            f"--minShapes=image_preprocessed:{min_batch_size}x3x{image_size}x{image_size}",
            f"--optShapes=image_preprocessed:{opt_batch_size}x3x{image_size}x{image_size}",
            f"--maxShapes=image_preprocessed:{max_batch_size}x3x{image_size}x{image_size}",
        ]

        if fp16_mode:
            args += ["--fp16"]
        
        subprocess.call(args)


class OwlVitImageEncoderTrt(object):
    def __init__(self, engine_path: str):

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        trt_module = TRTModule(
            engine=engine,
            input_names=["image_preprocessed"],
            output_names=["image_embeds"]
        )

        self.trt_module = trt_module

    @use_timer
    def __call__(self, pixel_values):
        return self.trt_module(pixel_values)


class OwlVitTextEncoderModule(nn.Module):
    def __init__(self, text_model, text_projection):
        super().__init__()
        self.text_model = text_model
        self.text_projection = text_projection

    @use_timer
    def forward(self, input_ids, attention_mask):
        text_outputs = self.text_model(input_ids, attention_mask)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        return text_embeds

    @staticmethod
    def from_pretrained(hf_name: str, device: str = "cuda"):
        hf_model = hf_build_model(hf_name)
        model = OwlVitTextEncoderModule(
            hf_model.owlvit.text_model, hf_model.owlvit.text_projection
        ).eval().to(device)
        return model

    @staticmethod
    def export_onnx(hf_name: str, output_path: str):
        
        model = OwlVitTextEncoderModule.from_pretrained(hf_name, "cpu")

        input_ids = torch.zeros(1, 16, dtype=torch.int64)
        attention_mask = torch.zeros(1, 16, dtype=torch.int64)

        dynamic_axes = {
            "input_ids": {0: "num_text"},
            "attention_mask": {0: "num_text"}
        }

        torch.onnx.export(
            model, 
            (input_ids, attention_mask), 
            output_path, 
            input_names=["input_ids", "attention_mask"], 
            output_names=["text_embeds"],
            dynamic_axes=dynamic_axes,
            opset_version=17
        )

    @staticmethod
    def build_trt(
            hf_name: str,
            output_path: str,
            min_num_text: int = 1,
            opt_num_text: int = 1,
            max_num_text: int = 20,
            onnx_path: str = None,
            fp16_mode: bool = True
        ):

        if onnx_path is None:
            onnx_path = tempfile.mktemp()
        
        if not os.path.exists(onnx_path):
            OwlVitTextEncoderModule.export_onnx(hf_name, onnx_path)

        trtexec = get_trtexec_executable()
        
        args = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={output_path}",
            f"--minShapes=input_ids:{min_num_text}x16,attention_mask:{min_num_text}x16",
            f"--optShapes=input_ids:{opt_num_text}x16,attention_mask:{opt_num_text}x16",
            f"--maxShapes=input_ids:{max_num_text}x16,attention_mask:{max_num_text}x16",
        ]

        # if fp16_mode:
        #     args += ["--fp16"]
        
        subprocess.call(args)


class OwlVitTextEncoderTrt(object):
    def __init__(self, engine_path: str):

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        trt_module = TRTModule(
            engine=engine,
            input_names=["input_ids", "attention_mask"],
            output_names=["text_embeds"]
        )

        self.trt_module = trt_module

    @use_timer
    def __call__(self, input_ids, attention_mask):
        return self.trt_module(input_ids, attention_mask)


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

    @use_timer
    def forward(self, image_embeds):
        pred_boxes = self.box_head(image_embeds)
        pred_boxes += self.box_bias
        pred_boxes = torch.sigmoid(pred_boxes)
        return pred_boxes


class OwlVitClassPredictorModule(nn.Module):
    def __init__(self, class_head):
        super().__init__()
        self.class_head = class_head

    @use_timer
    def forward(self, image_embeds, query_embeds, query_mask):

        pred_logits, class_embeds = self.class_head(
            image_embeds,
            query_embeds,
            query_mask
        )

        return pred_logits, class_embeds


class OwlVitDetectionPostprocessor(object):

    @use_timer
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
    def from_pretrained(
                name: str, 
                device: str = "cuda",
                image_encoder_engine = None,
                text_encoder_engine = None
            ):

        hf_model = hf_build_model(name)
        hf_processor = hf_build_processor(name)
        image_size = hf_get_image_size(name)
        patch_size = hf_get_patch_size(name)

        if image_encoder_engine is not None:
            if device != "cuda":
                raise RuntimeError("Device must be set to CUDA when using TensorRT.")
            image_encoder = OwlVitImageEncoderTrt(image_encoder_engine)
        else:
            image_encoder = OwlVitImageEncoderModule(
                hf_model.owlvit.vision_model, hf_model.layer_norm, image_size
            ).eval().to(device)

        if text_encoder_engine is not None:
            if device != "cuda":
                raise RuntimeError("Device must be set to CUDA when using TensorRT.")
            text_encoder = OwlVitTextEncoderTrt(text_encoder_engine)
        else:
            text_encoder = OwlVitTextEncoderModule(
                hf_model.owlvit.text_model, hf_model.owlvit.text_projection
            ).eval().to(device)


        return OwlVitPredictor(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            box_predictor=OwlVitBoxPredictorModule(
                    hf_model.box_head,
                    num_patches=image_size // patch_size
                ).eval().to(device),
            class_predictor=OwlVitClassPredictorModule(hf_model.class_head).eval().to(device),
            image_preprocessor=OwlVitImagePreprocessorModule(image_size).eval().to(device),
            image_formatter=OwlVitImageFormatter(device),
            text_tokenizer=OwlVitTextTokenizer(hf_processor, device),
            detection_postprocessor=OwlVitDetectionPostprocessor()
        )
    
    @use_timer
    def predict(self, image=None, text=None, threshold=0.1):
        #TODO: support multi-batch inference

        # Encode Text
        input_ids, attention_mask = self.text_tokenizer(text)

        text_embeds = self.text_encoder(input_ids, attention_mask)
        
        query_mask = input_ids[None, :, 0] > 0
        query_embeds = text_embeds[None, ...]

        # Encode Image
        image = self.image_formatter(image)
        image_preprocessed = self.image_preprocessor(image)
        image_embeds = self.image_encoder(image_preprocessed)

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

    