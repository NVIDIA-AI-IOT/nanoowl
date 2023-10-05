import pytest
import torch
import PIL.Image
from nanoowl.image_preprocessor import ImagePreprocessor


def test_image_preprocessor_preprocess_pil_image():

    image_preproc = ImagePreprocessor().to("cuda").eval()

    image = PIL.Image.open("assets/owl_glove_small.jpg")

    image_tensor = image_preproc.preprocess_pil_image(image)

    assert image_tensor.shape == (1, 3, 499, 756)
    assert torch.allclose(image_tensor.mean(), torch.zeros_like(image_tensor), atol=1, rtol=1)
