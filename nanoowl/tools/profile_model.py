from nanoowl.utils.module_recorder import ModuleRecorder
from nanoowl.utils.predictor import Predictor
import PIL.Image
import torch
import time

predictor = Predictor(
    vision_engine="data/owlvit_vision_model.engine"
    # vision_model_name="efficientvit_b0",
    # vision_checkpoint="data/efficientvit_b0_h8_ckpt.pth"
)

image = PIL.Image.open("assets/owl_glove.jpg")

output = predictor.predict_text(image, texts=["an owl", "a glove", "a face"])
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

    output = predictor.predict_text(image, texts=["an owl", "a glove", "a face"])

    for k, r in recorders.items():
        r.detach()
        results[k] += float(r.get_elapsed_time())

for k, r in results.items():
    print(f"{k}: {r / count}")


def profile_fps_module(module, data, count):
    t0 = time.perf_counter_ns()
    for i in range(count):
        output = module(*data)
        torch.cuda.current_stream().synchronize()
    t1 = time.perf_counter_ns()
    dt = (t1 - t0) / 1e9
    return count / dt

data = torch.randn(1,3,768,768).cuda()
a = torch.zeros(4, 16).cuda().long()
b = torch.zeros(4, 16).cuda()

fps_vis = profile_fps_module(predictor.model.owlvit.vision_model, [data], 50)
fps_tex = profile_fps_module(predictor.model.owlvit.text_model, [a, b], 50)
print(f"vis: {fps_vis}")
print(f"tex: {fps_tex}")