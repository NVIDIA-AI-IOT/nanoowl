import torch
import PIL.Image
import clip
from typing import List, Tuple, Optional
from nanoowl.image_preprocessor import ImagePreprocessor
from dataclasses import dataclass


__all__ = [
    "ClipPredictor"
]


@dataclass
class ClipEncodeTextOutput:
    text_embeds: torch.Tensor


@dataclass
class ClipEncodeImageOutput:
    image_embeds: torch.Tensor


@dataclass
class ClipDecodeOutput:
    labels: torch.Tensor
    scores: torch.Tensor


class ClipPredictor(torch.nn.Module):
    
    def __init__(self,
            model_name: str = "ViT-B/32",
            image_size: Tuple[int, int] = (224, 224),
            device: str = "cuda"
        ):
        super().__init__()
        self.device = device
        self.clip_model, _ = clip.load(model_name, device)
        self.image_size = image_size
    
    def get_image_size(self):
        return self.image_size

    def encode_text(self, text: List[str]) -> ClipEncodeTextOutput:
        text_tokens = clip.tokenize(text).to(self.device)
        text_embeds = self.clip_model.encode_text(text_tokens)
        return ClipEncodeTextOutput(text_embeds=text_embeds)

    def encode_image(self, image: torch.Tensor) -> ClipEncodeImageOutput:
        image_embeds = self.clip_model.encode_image(image)
        return ClipEncodeImageOutput(image_embeds=image_embeds)
    
    def decode(self, 
            image_output: ClipEncodeImageOutput, 
            text_output: ClipEncodeTextOutput
        ) -> ClipDecodeOutput:

        image_embeds = image_output.image_embeds
        text_embeds = text_output.text_embeds

        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        probs = torch.softmax(logits_per_image, dim=-1)
        prob_max = probs.max(dim=-1)
        
        return ClipDecodeOutput(
            labels=prob_max.indices,
            scores=prob_max.values
        )