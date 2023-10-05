FROM nvcr.io/nvidia/pytorch:23.01-py3

# upgrade pillow to fix "UnidentifiedImageError"
RUN pip install pillow --upgrade 

RUN pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git
RUN pip install transformers timm accelerate
RUN pip install git+https://github.com/openai/CLIP.git