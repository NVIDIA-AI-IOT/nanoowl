import pytest
import torch
import PIL.Image
from nanoowl.clip_predictor import ClipPredictor


def test_clip_preprocess_pil_image():

    clip_predictor = ClipPredictor()

    image = PIL.Image.open("assets/owl_glove_small.jpg")

    image_tensor = clip_predictor.preprocess_pil_image(image)

    assert image_tensor.shape == (1, 3, 499, 756)
    assert torch.allclose(image_tensor.mean(), torch.zeros_like(image_tensor), atol=1, rtol=1)


def test_clip_encode_text():

    clip_predictor = ClipPredictor()

    text_embeds = clip_predictor.encode_text(["a frog", "a dog"])

    assert text_embeds.shape == (2, 512)


def test_clip_encode_image():

    clip_predictor = ClipPredictor()

    image = PIL.Image.open("assets/owl_glove_small.jpg")

    image = image.resize((224, 224))

    image_tensor = clip_predictor.preprocess_pil_image(image)

    image_embeds = clip_predictor.encode_image(image_tensor)

    assert image_embeds.shape == (1, 512)


def test_export_image_encoder_onnx():
    pass


def test_build_image_encoder_engine():
    pass


def test_load_image_encoder_engine():
    pass


def test_image_size():
    clip_predictor = ClipPredictor()
    assert clip_predictor.get_image_size() == (224, 224)


def test_predict():
    pass