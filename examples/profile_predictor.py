import time
import PIL.Image
from nanoowl.utils.predictor import Predictor


image = PIL.Image.open("assets/owl_glove.jpg")
text = ["an owl", "a glove"]

predictor = Predictor(
    vision_engine="data/owlvit_vision_model.engine",
    threshold=0.1
)


count = 25

t0 = time.perf_counter_ns()
for i in range(count):
    output = predictor.predict(image=image, text=text)
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9

print(f"BOTH: {count/dt}")

t0 = time.perf_counter_ns()
for i in range(count):
    output = predictor.predict(text=text)
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9

print(f"TEXT: {count/dt}")

t0 = time.perf_counter_ns()
for i in range(count):
    output = predictor.predict(image=image)
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9

print(f"VISION: {count/dt}")