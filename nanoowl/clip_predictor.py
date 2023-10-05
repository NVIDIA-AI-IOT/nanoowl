import torch
import PIL.Image
from typing import List


__all__ = [
    "ClipPredictor"
]


class ClipPredictor(object):
    def encode_text(self, text: List[str]):
        pass

    def encode_image(self, image: torch.Tensor):
        pass
    
    def export_image_encoder_onnx(self, output_path: str):
        pass
    
    def build_image_encoder_engine(self, onnx_path: str, output_path: str):
        pass
    
    def load_image_encoder_engine(self, engine_path: str):
        pass
    
    def image_size(self):
        pass

    def predict(self, image: PIL.Image, text: List[str]):
        pass