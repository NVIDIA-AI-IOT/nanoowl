from .predictor import Predictor
from .graph import Graph, Op
import PIL.Image
import torch
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GraphDetection:
    box: Tuple[float, float, float, float]
    instance_id: int
    parent_instance_id: int
    labels: List[int]
    scores: List[float]


class GraphPredictor:
    def __init__(self,
            predictor: Predictor
        ):
        self.predictor = predictor
        self.graph = None
        self._owl_text_encodes = {}
        self._clip_text_encodes = {}

    def _compute_owl_text_encodes(self, graph):
        text_encodes = {}
        for node in graph.nodes:
            if node.op == Op.DETECT:
                text_encodes[node.id] = self.predictor.owl_encode_text(node.labels)
        return text_encodes

    def _compute_clip_text_encodes(self, graph):
        text_encodes = {}
        for node in graph.nodes:
            if node.op == Op.CLASSIFY:
                text_encodes[node.id] = self.predictor.clip_encode_text(node.labels)
        return text_encodes

    def set_graph(self, graph: Graph):
        self.graph = graph
        self._owl_text_encodes = self._compute_owl_text_encodes(graph)
        self._clip_text_encodes = self._compute_clip_text_encodes(graph)

    def set_prompt(self, prompt: str):
        self.set_graph(Graph.from_prompt(prompt))

    def predict(self, image: PIL.Image, threshold=0.1, starting_rois=None):
        graph = self.graph

        image_tensor = self.predictor.preprocess_image(image)
        if starting_rois is None:
            starting_rois = torch.tensor([[0, 0, image.width, image.height]], dtype=image_tensor.dtype, device=image_tensor.device)
        else:
            starting_rois = torch.tensor(starting_rois, dtype=image_tensor.dtype, device=image_tensor.device)

        boxes = {
            0: starting_rois
        }
        scores = {
            0: torch.tensor([1.], dtype=torch.float, device=image_tensor.device).repeat(len(starting_rois))
        }
        instance_ids = {
            0: torch.arange(len(starting_rois), dtype=torch.int64, device=image_tensor.device)
        }
        parent_instance_ids = {
            0: torch.tensor([-1], dtype=torch.int64, device=image_tensor.device).repeat(len(starting_rois))
        }

        node_owl_outputs = {}
        node_clip_outputs = {}
        owl_image_encodings = {}
        clip_image_encodings = {}

        global_instance_id = len(starting_rois)

        queue = [0]

        while queue:
            cur_buf = queue.pop(0)

            detect_nodes = graph.find_nodes(input_buffer=cur_buf, op=Op.DETECT)
            if len(detect_nodes) > 0 and cur_buf not in owl_image_encodings:
                # TODO: use cached parent value for classify
                owl_image_encodings[cur_buf] = self.predictor.owl_encode_rois(image_tensor, boxes[cur_buf])

            classify_nodes = graph.find_nodes(input_buffer=cur_buf, op=Op.CLASSIFY)
            if len(classify_nodes) > 0 and cur_buf not in clip_image_encodings:
                #TODO: use cached parent value for classify
                clip_image_encodings[cur_buf] = self.predictor.clip_encode_rois(image_tensor, boxes[cur_buf])

            for node in detect_nodes:
                text_encodings = self._owl_text_encodes[node.id]
                node_input = owl_image_encodings[node.input_buffer]
                node_output = self.predictor.owl_decode(node_input, [text_encodings]*len(node_input.logit_shift), threshold=threshold)
                node_owl_outputs[node.id] = node_output

                num_detections = len(node_output.labels)
                instance_ids_for_node = torch.arange(global_instance_id, global_instance_id + num_detections, dtype=torch.int64, device=node_output.labels.device)
                parent_instance_ids_for_node = instance_ids[node.input_buffer][node_output.roi_indices]
                global_instance_id += num_detections

                for i in range(len(node.labels)):
                    mask = node_output.labels == i
                    out_idx = node.output_buffers[i]
                    boxes[out_idx] = node_output.boxes[mask]
                    scores[out_idx] = node_output.scores[mask]
                    instance_ids[out_idx] = instance_ids_for_node[mask]
                    parent_instance_ids[out_idx] = parent_instance_ids_for_node[mask]

            for node in classify_nodes:
                text_encodings = self._clip_text_encodes[node.id]
                node_input = clip_image_encodings[node.input_buffer]
                node_output = self.predictor.clip_decode(node_input, text_encodings)
                node_clip_outputs[node.id] = node_output
                parent_instance_ids_for_node = instance_ids[node.input_buffer]

                for i in range(len(node.labels)):
                    mask = node_output.labels == i
                    output_buffer = node.output_buffers[i]
                    scores[output_buffer] = node_output.scores[mask].float()
                    boxes[output_buffer] = boxes[cur_buf][mask].float()
                    instance_ids[output_buffer] = instance_ids[node.input_buffer][mask]
                    parent_instance_ids[output_buffer] = parent_instance_ids[node.input_buffer][mask]

            for node in detect_nodes:
                for buf in node.output_buffers:
                    if buf in scores and len(scores[buf]) > 0:
                        queue.append(buf)

            for node in classify_nodes:
                for buf in node.output_buffers:
                    if buf in scores and len(scores[buf]) > 0:
                        queue.append(buf)

        # group boxes and add all labels
        detections = {}
        for i in boxes.keys():
            for box, score, instance_id, parent_instance_id in zip(boxes[i], scores[i], instance_ids[i], parent_instance_ids[i]):
                instance_id = int(instance_id)
                score = float(score)
                box = box.tolist()
                parent_instance_id = int(parent_instance_id)
                if instance_id in detections:
                    detections[instance_id].labels.append(i)
                    detections[instance_id].scores.append(score)
                else:
                    detections[instance_id] = GraphDetection(
                        box=box,
                        instance_id=instance_id,
                        parent_instance_id=parent_instance_id,
                        scores=[score],
                        labels=[i]
                    )
                

        return detections