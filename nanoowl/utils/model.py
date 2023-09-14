import torch
from transformers import (
    OwlViTForObjectDetection
)
from nanoowl.utils.tensorrt import load_image_encoder_engine
from nanoowl.utils.transform import build_owlvit_vision_transform
from nanoowl.models import create_model


def load_owlvit_model(
        device="cuda", 
        vision_engine=None,
        vision_checkpoint=None,
        vision_model_name=None
    ):
    
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", device_map=device)

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

