import torch
import PIL.Image
import clip
from typing import List, Tuple, Optional
from nanoowl.image_preprocessor import ImagePreprocessor


__all__ = [
    "ClipPredictor"
]


class ClipPredictor(object):
    
    def __init__(self,
            model_name: str = "ViT-B/32",
            image_size: Tuple[int, int] = (224, 224),
            image_preprocessor: Optional[ImagePreprocessor] = None,
            device: str = "cuda"
        ):
        super().__init__()

        self.device = device
        
        if image_preprocessor is None:
            image_preprocessor = ImagePreprocessor()

        self.image_preprocessor = image_preprocessor.to(device)

        self.clip_model, _ = clip.load(model_name, device)
        self.image_size = image_size
    
    def preprocess_pil_image(self, image: PIL.Image.Image):
        return self.image_preprocessor.preprocess_pil_image(image)

    def encode_text(self, text: List[str]):
        text_tokens = clip.tokenize(text).to(self.device)
        text_embeds = self.clip_model.encode_text(text_tokens)
        return text_embeds

    def encode_image(self, image: torch.Tensor):
        image_embeds = self.clip_model.encode_image(image)
        return image_embeds
    
    def export_image_encoder_onnx(self, output_path: str):
        raise NotImplementedError
    
    def build_image_encoder_engine(self, onnx_path: str, output_path: str):
        raise NotImplementedError
    
    def load_image_encoder_engine(self, engine_path: str):
        raise NotImplementedError
    
    def get_image_size(self):
        return self.image_size

    def predict(self, image: PIL.Image.Image, text: List[str]):
        raise NotImplementedError