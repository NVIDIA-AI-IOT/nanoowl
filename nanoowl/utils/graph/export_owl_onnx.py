from .predictor import Predictor


predictor = Predictor()

predictor.export_owl_image_encoder_onnx("data/owl_image_encoder.onnx", use_dynamic_axes=True)