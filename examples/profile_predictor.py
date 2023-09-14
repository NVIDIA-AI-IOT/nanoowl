import time
import PIL.Image
import torch
from nanoowl.utils.predictor import Predictor, remap_device


image = PIL.Image.open("assets/owl_glove.jpg")
text = ["an owl", "a glove"]

predictor = Predictor(
    vision_engine="data/owlvit_vision_model.engine",
    threshold=0.1
)

count = 50
predictor.set_image(image)
for i in range(5):
    output = predictor.predict(image=image, text=text)


torch.cuda.current_stream().synchronize()
t0 = time.perf_counter_ns()
for i in range(count):
    # predictor.set_image(image)
    outputs = predictor._run_model()
    # outputs = remap_device(outputs, "cpu")
    # target_sizes = torch.Tensor([image.size[::-1]])
    # results = predictor.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=predictor.threshold)
    # i = 0
    # boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    # detections = []
    # for box, score, label in zip(boxes, scores, labels):
    #     detection = {"bbox": box.tolist(), "score": float(score), "label": int(label), "text": text[label]}
    #     detections.append(detection)
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

print(f"PREDICT IMAGE: {count/dt}")

torch.cuda.current_stream().synchronize()
t0 = time.perf_counter_ns()
for i in range(count):
    output = predictor.set_image(image=image)
    torch.cuda.current_stream().synchronize()
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9

print(f"SET IMAGE: {count/dt}")

torch.cuda.current_stream().synchronize()
t0 = time.perf_counter_ns()
for i in range(count):
    box_preds = predictor.box_predictor(predictor._image_embeds, predictor._feature_map)
    torch.cuda.current_stream().synchronize()
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9

print(f"PRED BOXES: {count/dt}")

x = torch.randn(1, 3, 768, 768).cuda()

torch.cuda.current_stream().synchronize()
t0 = time.perf_counter_ns()
for i in range(count):
    output = predictor.model.owlvit.vision_model(pixel_values=x)
    torch.cuda.current_stream().synchronize()
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9

print(f"VISION RAW: {count/dt}")


torch.cuda.current_stream().synchronize()
t0 = time.perf_counter_ns()
for i in range(count):
    output = predictor._transform(image)
    torch.cuda.current_stream().synchronize()
t1 = time.perf_counter_ns()
dt = (t1 - t0) / 1e9

print(f"TRANSFORM: {count/dt}")