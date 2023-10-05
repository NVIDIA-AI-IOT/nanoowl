import torch
import PIL.Image
from typing import List, Tuple, Optional
from nanoowl.image_preprocessor import ImagePreprocessor


__all__ = [
    "ClipPredictor"
]


class ClipPredictor(torch.nn.Module):
    
    def __init__(self,
            image_preprocessor: Optional[ImagePreprocessor] = None
        ):
        super().__init__()
        
        if image_preprocessor is None:
            image_preprocessor = ImagePreprocessor()

        self.image_preprocessor = image_preprocessor
        
    def preprocess_pil_image(self, image: PIL.Image.Image):
        return self.image_preprocessor.preprocess_pil_image(image)

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

    def predict(self, image: PIL.Image.Image, text: List[str]):
        pass