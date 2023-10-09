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


import json
from enum import Enum
from typing import List, Optional, Mapping
from .clip_predictor import ClipEncodeTextOutput
from .owl_predictor import OwlEncodeTextOutput


__all__ = [
    "TreeOp",
    "TreeNode",
    "Tree"
]


class TreeOp(Enum):
    DETECT = "detect"
    CLASSIFY = "classify"

    def __str__(self) -> str:
        return str(self.value)


class TreeNode:
    op: TreeOp
    input: int
    outputs: List[int]

    def __init__(self, op: TreeOp, input: int, outputs: Optional[List[int]] = None):
        self.op = op
        self.input = input
        self.outputs = [] if outputs is None else outputs

    def to_dict(self):
        return {
            "op": str(self.op),
            "input": self.input,
            "outputs": self.outputs
        }

    @staticmethod
    def from_dict(node_dict: dict):

        if "op" not in node_dict:
            raise RuntimeError("Missing 'op' field.")

        if "input" not in node_dict:
            raise RuntimeError("Missing 'input' field.")
        
        if "outputs" not in node_dict:
            raise RuntimeError("Missing 'input' field.")
        
        return TreeNode(
            op=node_dict["op"],
            input=node_dict["input"],
            outputs=node_dict["outputs"]
        )
    

class Tree:
    nodes: List[TreeNode]
    labels: List[str]

    def __init__(self, nodes, labels):
        self.nodes = nodes
        self.labels = labels
        self._label_index_to_node_map = self._build_label_index_to_node_map()
    
    def _build_label_index_to_node_map(self) -> Mapping[int, "TreeNode"]:
        label_to_node_map = {}
        for node in self.nodes:
            for label_index in node.outputs:
                if label_index in label_to_node_map:
                    raise RuntimeError("Duplicate output label.")
                label_to_node_map[label_index] = node
        return label_to_node_map
    
    def to_dict(self):
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "labels": self.labels
        }

    @staticmethod
    def from_prompt(prompt: str):

        nodes = []
        node_stack = []
        label_index_stack = [0]
        labels = ["image"]
        label_index = 0

        for ch in prompt:

            if ch == "[":
                label_index += 1
                node = TreeNode(op=TreeOp.DETECT, input=label_index_stack[-1])
                node.outputs.append(label_index)
                node_stack.append(node)
                label_index_stack.append(label_index)
                labels.append("")
                nodes.append(node)
            elif ch == "]":
                if len(node_stack) == 0:
                    raise RuntimeError("Unexpected ']'.")
                node = node_stack.pop()
                if node.op != TreeOp.DETECT:
                    raise RuntimeError("Unexpected ']'.")
                label_index_stack.pop()
            elif ch == "(":
                label_index = label_index + 1
                node = TreeNode(op=TreeOp.CLASSIFY, input=label_index_stack[-1])
                node.outputs.append(label_index)
                node_stack.append(node)
                label_index_stack.append(label_index)
                labels.append("")
                nodes.append(node)
            elif ch == ")":
                if len(node_stack) == 0:
                    raise RuntimeError("Unexpected ')'.")
                node = node_stack.pop()
                if node.op != TreeOp.CLASSIFY:
                    raise RuntimeError("Unexpected ')'.")
                label_index_stack.pop()
            elif ch == ",":
                label_index_stack.pop()
                label_index = label_index + 1
                label_index_stack.append(label_index)
                node_stack[-1].outputs.append(label_index)
                labels.append("")
            else:
                labels[label_index_stack[-1]] += ch

        if len(node_stack) > 0:
            if node_stack[-1].op == TreeOp.DETECT:
                raise RuntimeError("Missing ']'.")
            if node_stack[-1].op == TreeOp.CLASSIFY:
                raise RuntimeError("Missing ')'.")
            
        labels = [label.strip() for label in labels]

        graph = Tree(nodes=nodes, labels=labels)

        return graph
    
    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @staticmethod
    def from_dict(tree_dict: dict) -> "Tree":

        if "nodes" not in tree_dict:
            raise RuntimeError("Missing 'nodes' field.")
        
        if "labels" not in tree_dict:
            raise RuntimeError("Missing 'labels' field.")
        
        nodes = [TreeNode.from_dict(node_dict) for node_dict in tree_dict["nodes"]]
        labels = tree_dict["labels"]

        return Tree(nodes=nodes, labels=labels)

    @staticmethod
    def from_json(tree_json: str) -> "Tree":
        tree_dict = json.loads(tree_json)
        return Tree.from_dict(tree_dict)
    
    def get_op_for_label_index(self, label_index: int):
        if label_index not in self._label_index_to_node_map:
            return None
        return self._label_index_to_node_map[label_index].op
    
    def get_label_indices_with_op(self, op: TreeOp):
        return [
            index for index in range(len(self.labels))
            if self.get_op_for_label_index(index) == op
        ]
    
    def get_classify_label_indices(self):
        return self.get_label_indices_with_op(TreeOp.CLASSIFY)
    
    def get_detect_label_indices(self):
        return self.get_label_indices_with_op(TreeOp.DETECT)

    def find_nodes_with_input(self, input_index: int):
        return [n for n in self.nodes if n.input == input_index]

    def find_detect_nodes_with_input(self, input_index: int):
        return [n for n in self.find_nodes_with_input(input_index) if n.op == TreeOp.DETECT]

    def find_classify_nodes_with_input(self, input_index: int):
        return [n for n in self.find_nodes_with_input(input_index) if n.op == TreeOp.CLASSIFY]

    def get_label_depth(self, index):
        depth = 0
        while index in self._label_index_to_node_map:
            depth += 1
            node = self._label_index_to_node_map[index]
            index = node.input
        return depth

    def get_label_depth_map(self):
        depths = {}
        for i in range(len(self.labels)):
            depths[i] = self.get_label_depth(i)
        return depths

    def get_label_map(self):
        label_map = {}
        for i in range(len(self.labels)):
            label_map[i] = self.labels[i]
        return label_map