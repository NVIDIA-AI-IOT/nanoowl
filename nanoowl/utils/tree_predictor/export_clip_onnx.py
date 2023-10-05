from .model import TreeModel


model = TreeModel()

model.export_clip_image_encoder_onnx("data/clip_image_encoder.onnx", use_dynamic_axes=True)