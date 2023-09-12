# NanoOWL

<p align="center"><a href="#usage"/>👍 Usage</a> - <a href="#performance"/>⏱️ Performance</a> - <a href="#setup">🛠️ Setup</a> - <a href="#examples">🤸 Examples</a> - <a href="#acknowledgement">👏 Acknowledgment</a> - <a href="#see-also">🔗 See also</a></p>

NanoOWL is a project that optimizes [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) to run 🔥 ***real-time*** 🔥 on [NVIDIA Jetson AGX Orin](https://store.nvidia.com/en-us/jetson/store) with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).  

> Interested in detecting object masks as well?  Try combining NanoOWL with
> [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam) for zero-shot open-vocabulary 
> instance segmentation.

<a id="usage"></a>
## 👍 Usage

You can use NanoOWL in Python like this

```python3
from nanoowl.utils.predictor import Predictor

predictor = Predictor(
    vision_engine="data/owlvit_vision_model.engine",
    tresh=0.1
)

image = PIL.Image.open("assets/dogs.jpg")

detectors = predictor.predict(image, texts=["a dog"])
```

<a id="performance"></a>
## ⏱️ Performance

NanoOWL runs real-time on Jetson AGX Orin.

<table style="border-top: solid 1px; border-left: solid 1px; border-right: solid 1px; border-bottom: solid 1px">
    <thead>
        <tr>
            <th rowspan=1 style="text-align: center; border-right: solid 1px">Model †</th>
            <th colspan=1 style="text-align: center; border-right: solid 1px">⏱️ Jetson AGX Orin (ms)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align: center; border-right: solid 1px">OWL-ViT (transformers default, CPU)</td>
            <td style="text-align: center; border-right: solid 1px">950</td>
        </tr>
        <tr>
            <td style="text-align: center; border-right: solid 1px">NanoOWL (OWL-ViT GPU+TRT optimized)</td>
            <td style="text-align: center; border-right: solid 1px">45</td>
        </tr>
    </tbody>
</table>

<a id="setup"></a>
## 🛠️ Setup

1. Install the dependencies

    1. Install PyTorch

    2. Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
    3. Install NVIDIA TensorRT
    4. Install the Transformers library

        ```bash
        python3 -m pip install transformers
        ```
    5. (optional) Install NanoSAM (for the instance segmentation example)

2. Install the NanoOWL package.

    ```bash
    git clone https://github.com/NVIDIA-AI-IOT/nanosam
    cd nanosam
    python3 setup.py develop --user
    ```

3. Build the TensorRT engine for the OWL-ViT vision encoder

    1. Export the OWL-ViT vision encoder to ONNX

        ```bash
        python3 -m nanoowl.tools.export_vision_model_onnx \
            --output="data/owlvit_vision_model.onnx"
        ```
    
    2. Build the TensorRT engine with ``trtexec``

        ```bash
        trtexec \
            --onnx=data/owlvit_vision_model.onnx \
            --saveEngine=data/owlvit_vision_model.engine \
            --fp16
        ```

4. Run the basic usage example to ensure everything is working

    ```bash
    python3 examples/basic_usage.py \
        --vision_engine=data/owlvit_vision_model.engine \
        --thresh=0.1
    ```

That's it!  If everything is working properly, you should see a visualization saved
to ``data/visualize_owlvit_out.jpg``.  

<a id="examples"></a>
## 🤸 Examples

### Example 1 - Basic usage

### Example 2 - Live camera demo

### Example 3 - Instance segmentation with [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam)


<a id="acknowledgement"></a>
## 👏 Acknowledgement

Thanks to the authors of [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) for the great open-vocabluary detection work.

<a id="see-also"></a>
## 🔗 See also

- [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam) - A real-time Segment Anything (SAM) model variant for NVIDIA Jetson Orin platforms.
- [Jetson Introduction to Knowledge Distillation Tutorial](https://github.com/NVIDIA-AI-IOT/jetson-intro-to-distillation) - For an introduction to knowledge distillation as a model optimization technique.
- [Jetson Generative AI Playground](https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/) - For instructions and tips for using a variety of LLMs and transformers on Jetson.
- [Jetson Containers](https://github.com/dusty-nv/jetson-containers) - For a variety of easily deployable and modular Jetson Containers
