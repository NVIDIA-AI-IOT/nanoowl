import pytest
import torch
import PIL.Image
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.image_preprocessor import ImagePreprocessor


def test_owl_predictor_get_image_size():
    owl_predictor = OwlPredictor()
    assert owl_predictor.get_image_size() == (768, 768)


def test_owl_predictor_encode_text():

    owl_predictor = OwlPredictor()

    text_encode_output = owl_predictor.encode_text(["a frog", "a dog"])

    assert text_encode_output.text_embeds.shape == (2, 512)


def test_owl_predictor_encode_image():

    owl_predictor = OwlPredictor()

    image = PIL.Image.open("assets/owl_glove_small.jpg")

    image = image.resize(owl_predictor.get_image_size())

    image_preprocessor = ImagePreprocessor().to(owl_predictor.device).eval()

    image_tensor = image_preprocessor.preprocess_pil_image(image)

    image_encode_output = owl_predictor.encode_image(image_tensor)

    assert image_encode_output.image_class_embeds.shape == (1, owl_predictor.get_num_patches(), 512)


def test_owl_predictor_decode():

    owl_predictor = OwlPredictor()

    image = PIL.Image.open("assets/owl_glove_small.jpg")

    image = image.resize(owl_predictor.get_image_size())
    image_preprocessor = ImagePreprocessor().to(owl_predictor.device).eval()

    image_tensor = image_preprocessor.preprocess_pil_image(image)

    text_output = owl_predictor.encode_text(["an owl"])
    image_output = owl_predictor.encode_image(image_tensor)

    classify_output = owl_predictor.decode(image_output, text_output)

    assert len(classify_output.labels == 1)
    assert classify_output.labels[0] == 0
    assert classify_output.boxes.shape == (1, 4)
    assert classify_output.input_indices == 0


def test_owl_predictor_decode_multiple_images():

    owl_predictor = OwlPredictor()
    image_preprocessor = ImagePreprocessor().to(owl_predictor.device).eval()

    image_paths = [
        "assets/owl_glove_small.jpg",
        "assets/frog.jpg"
    ]

    images = []
    for image_path in image_paths:
        image = PIL.Image.open(image_path)
        image = image.resize(owl_predictor.get_image_size())
        image = image_preprocessor.preprocess_pil_image(image)
        images.append(image)

    images = torch.cat(images, dim=0)

    text_output = owl_predictor.encode_text(["an owl", "a frog"])
    image_output = owl_predictor.encode_image(images)

    decode_output = owl_predictor.decode(image_output, text_output)

    # check number of detections
    assert len(decode_output.labels == 2)

    # check owl detection
    assert decode_output.labels[0] == 0
    assert decode_output.input_indices[0] == 0
    
    # check frog detection
    assert decode_output.labels[1] == 1
    assert decode_output.input_indices[1] == 1
