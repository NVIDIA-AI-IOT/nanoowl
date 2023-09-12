from nanoowl.utils.module_recorder import ModuleRecorder
from nanoowl.utils.predictor import Predictor
import PIL.Image
import torch

predictor = Predictor(
    vision_engine="data/owlvit_vision_model.engine"
    # vision_model_name="efficientvit_b0",
    # vision_checkpoint="data/efficientvit_b0_h8_ckpt.pth"
)

image = PIL.Image.open("assets/owl_glove.jpg")

output = predictor.predict(image, texts=["an owl", "a glove", "a face"])
torch.cuda.current_stream().synchronize()

recorders = {
    "model": ModuleRecorder(predictor.model),
    "model.owlvit": ModuleRecorder(predictor.model.owlvit),
    "model.owlvit.text_model": ModuleRecorder(predictor.model.owlvit.text_model),
    "model.owlvit.vision_model": ModuleRecorder(predictor.model.owlvit.vision_model),
}

count = 100

results = {k: 0. for k in recorders.keys()}

for i in range(count):
    for k, r in recorders.items():
        r.attach()

    output = predictor.predict(image, texts=["an owl", "a glove", "a face"])

    for k, r in recorders.items():
        r.detach()
        results[k] += float(r.get_elapsed_time())

for k, r in results.items():
    print(f"{k}: {r / count}")