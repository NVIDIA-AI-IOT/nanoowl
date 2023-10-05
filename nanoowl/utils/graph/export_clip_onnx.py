from .predictor import Predictor


predictor = Predictor()

predictor.export_clip_image_encoder_onnx("data/clip_image_encoder.onnx", use_dynamic_axes=True)