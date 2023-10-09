from .owl_predictor import OwlPredictor


predictor = OwlPredictor()
predictor.export_image_encoder_onnx("data/owl_image_encoder.onnx")