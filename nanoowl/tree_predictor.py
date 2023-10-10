# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .tree import Tree, TreeOp
from .owl_predictor import OwlPredictor, OwlEncodeTextOutput, OwlEncodeImageOutput
from .clip_predictor import ClipPredictor, ClipEncodeTextOutput, ClipEncodeImageOutput
from .image_preprocessor import ImagePreprocessor

import torch
import PIL.Image
from typing import Optional, Tuple, List, Mapping, Dict
from dataclasses import dataclass


@dataclass
class TreeDetection:
    id: int
    parent_id: int
    box: Tuple[float, float, float, float]
    labels: List[int]
    scores: List[int]


@dataclass
class TreeOutput:
    detections: List[TreeDetection]
    

class TreePredictor(torch.nn.Module):

    def __init__(self,
            owl_predictor: Optional[OwlPredictor] = None,
            clip_predictor: Optional[ClipPredictor] = None,
            image_preprocessor: Optional[ImagePreprocessor] = None,
            device: str = "cuda"
        ):
        super().__init__()
        self.owl_predictor = OwlPredictor() if owl_predictor is None else owl_predictor
        self.clip_predictor = ClipPredictor() if clip_predictor is None else clip_predictor
        self.image_preprocessor = ImagePreprocessor().to(device).eval() if image_preprocessor is None else image_preprocessor

    def encode_clip_text(self, tree: Tree) -> Dict[int, ClipEncodeTextOutput]:
        label_indices = tree.get_classify_label_indices()
        if len(label_indices) == 0:
            return {}
        labels = [tree.labels[index] for index in label_indices]
        text_encodings = self.clip_predictor.encode_text(labels)
        label_encodings = {}
        for i in range(len(labels)):
            label_encodings[label_indices[i]] = text_encodings.slice(i, i+1)
        return label_encodings
    
    def encode_owl_text(self, tree: Tree) -> Dict[int, OwlEncodeTextOutput]:
        label_indices = tree.get_detect_label_indices()
        if len(label_indices) == 0:
            return {}
        labels = [tree.labels[index] for index in label_indices]
        text_encodings = self.owl_predictor.encode_text(labels)
        label_encodings = {}
        for i in range(len(labels)):
            label_encodings[label_indices[i]] = text_encodings.slice(i, i+1)
        return label_encodings
    
    @torch.no_grad()
    def predict(self, 
            image: PIL.Image.Image, 
            tree: Tree, 
            threshold: float = 0.1,
            clip_text_encodings: Optional[Dict[int, ClipEncodeTextOutput]] = None,
            owl_text_encodings: Optional[Dict[int, OwlEncodeTextOutput]] = None
        ):

        if clip_text_encodings is None:
            clip_text_encodings = self.encode_clip_text(tree)
        
        if owl_text_encodings is None:
            owl_text_encodings = self.encode_owl_text(tree)
        
        image_tensor = self.image_preprocessor.preprocess_pil_image(image)
        boxes = {
            0: torch.tensor([[0, 0, image.width, image.height]], dtype=image_tensor.dtype, device=image_tensor.device)
        }
        scores = {
            0: torch.tensor([1.], dtype=torch.float, device=image_tensor.device)
        }
        instance_ids = {
            0: torch.tensor([0], dtype=torch.int64, device=image_tensor.device)
        }
        parent_instance_ids = {
            0: torch.tensor([-1], dtype=torch.int64, device=image_tensor.device)
        }

        owl_image_encodings: Dict[int, OwlEncodeImageOutput] = {}
        clip_image_encodings: Dict[int, ClipEncodeImageOutput] = {}

        global_instance_id = 1

        queue = [0]

        while queue:
            label_index = queue.pop(0)

            detect_nodes = tree.find_detect_nodes_with_input(label_index)
            classify_nodes = tree.find_classify_nodes_with_input(label_index)

            # Run OWL image encode if required
            if len(detect_nodes) > 0 and label_index not in owl_image_encodings:
                owl_image_encodings[label_index] = self.owl_predictor.encode_rois(image_tensor, boxes[label_index])
                

            # Run CLIP image encode if required
            if len(classify_nodes) > 0 and label_index not in clip_image_encodings:
                clip_image_encodings[label_index] = self.clip_predictor.encode_rois(image_tensor, boxes[label_index])

            # Decode detect nodes
            for node in detect_nodes:

                if node.input not in owl_image_encodings:
                    raise RuntimeError("Missing owl image encodings for node.")

                # gather encodings
                owl_text_encodings_for_node = OwlEncodeTextOutput(
                    text_embeds=torch.cat([
                        owl_text_encodings[i].text_embeds for i in node.outputs
                    ], dim=0)
                )
                
                owl_node_output = self.owl_predictor.decode(
                    owl_image_encodings[node.input], 
                    owl_text_encodings_for_node, 
                    threshold=threshold
                )

                num_detections = len(owl_node_output.labels)
                instance_ids_for_node = torch.arange(global_instance_id, global_instance_id + num_detections, dtype=torch.int64, device=owl_node_output.labels.device)
                parent_instance_ids_for_node = instance_ids[node.input][owl_node_output.input_indices]
                global_instance_id += num_detections

                for i in range(len(node.outputs)):
                    mask = owl_node_output.labels == i
                    out_idx = node.outputs[i]
                    boxes[out_idx] = owl_node_output.boxes[mask]
                    scores[out_idx] = owl_node_output.scores[mask]
                    instance_ids[out_idx] = instance_ids_for_node[mask]
                    parent_instance_ids[out_idx] = parent_instance_ids_for_node[mask]

            for node in classify_nodes:

                if node.input not in clip_image_encodings:
                    raise RuntimeError("Missing clip image encodings for node.")

                clip_text_encodings_for_node = ClipEncodeTextOutput(
                    text_embeds=torch.cat([
                        clip_text_encodings[i].text_embeds for i in node.outputs
                    ], dim=0)
                )

                clip_node_output = self.clip_predictor.decode(
                    clip_image_encodings[node.input], 
                    clip_text_encodings_for_node
                )

                parent_instance_ids_for_node = instance_ids[node.input]

                for i in range(len(node.outputs)):
                    mask = clip_node_output.labels == i
                    output_buffer = node.outputs[i]
                    scores[output_buffer] = clip_node_output.scores[mask].float()
                    boxes[output_buffer] = boxes[label_index][mask].float()
                    instance_ids[output_buffer] = instance_ids[node.input][mask]
                    parent_instance_ids[output_buffer] = parent_instance_ids[node.input][mask]

            for node in detect_nodes:
                for buf in node.outputs:
                    if buf in scores and len(scores[buf]) > 0:
                        queue.append(buf)

            for node in classify_nodes:
                for buf in node.outputs:
                    if buf in scores and len(scores[buf]) > 0:
                        queue.append(buf)

        # Fill outputs
        detections: Dict[int, TreeDetection] = {}
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
                    detections[instance_id] = TreeDetection(
                        id=instance_id,
                        parent_id=parent_instance_id,
                        box=box,
                        labels=[i],
                        scores=[score]
                    )

        return TreeOutput(detections=detections.values())