import time
import PIL.Image
import torch
from nanoowl.utils.predictor import Predictor


image = PIL.Image.open("assets/owl_glove.jpg")
text = ["an owl", "a glove"]

predictor = Predictor(
    # vision_engine="data/owlvit_vision_model.engine",
    threshold=0.1
)

count = 50
predictor.set_image(image)
for i in range(5):
    output = predictor.predict(image=image, text=text)


torch.cuda.current_stream().synchronize()
t0 = time.perf_counter_ns()
for i in range(count):
    output = predictor.predict()
    torch.cuda.current_stream().synchronize()
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9

print(f"RUN MODEL: {count/dt}")

torch.cuda.current_stream().synchronize()
t0 = time.perf_counter_ns()
for i in range(count):
    output = predictor.predict(text=text)
    torch.cuda.current_stream().synchronize()
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9

print(f"TEXT: {count/dt}")

torch.cuda.current_stream().synchronize()
t0 = time.perf_counter_ns()
for i in range(count):
    output = predictor.predict(image=image)
    torch.cuda.current_stream().synchronize()
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9

print(f"VISION: {count/dt}")