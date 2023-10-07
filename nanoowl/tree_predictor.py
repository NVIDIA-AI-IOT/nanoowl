import torch
import PIL.Image
from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from .clip_predictor import ClipPredictor
from .owl_predictor import OwlPredictor


__all__ = [
    "TreePredictor"
]



class OpType(Enum):
    DETECT = "DETECT"
    CLASSIFY = "CLASSIFY"


@dataclass
class Topic:
    graph: "Graph"
    id: int
    name: str


@dataclass
class Node:
    graph: "Graph"
    id: int
    input_topics: List[int]
    output_topics: List[int]


@dataclass
class LabelsNode(Node):
    labels: List[str]

    def new_label(self):
        topic = self.graph.new_topic("")
        self.labels.append("")
        self.output_topics.append(topic)

    def append_char_to_label(self, char: str):
        self.labels

@dataclass
class DetectNode(Node):
    labels: List[str]

@dataclass
class 

class Graph:

    def __init__(self):
        self.nodes = []
        self.topics = []
        
    def new_node(self, op: Op, input_topics: List[int], output_topics: List[int]) -> Node:
        node = Node(
            graph=self,
            id=len(self.nodes), 
            op=op, 
            input_topics=input_topics, 
            output_topics=output_topics
        )
        self.nodes.append(node)
        return node

    def get_node(self, id: int) -> Node:
        return self.nodes[id]

    def new_topic(self, name: str) -> Topic:
        topic = Topic(
            graph=self,
            id=len(self.topics), 
            name=name
        )
        self.topics.append(topic)
        return topic

    def get_topic(self, id: int) -> Topic:
        return self.topics[id]

    @staticmethod
    def parse_tree_prompt(prompt: str):

        graph = Graph()
        topic = graph.add_topic("input")

        for ch in prompt:
            if ch == "[":
                g
            elif ch == "]":
                pass
            elif ch == "(":
                pass
            elif ch == ")":
                pass
            elif ch == ",":
                pass
            else:
                pass
        pass

# class TreePredictor(object):
#     def __init__(self,
#             clip_predictor: ClipPredictor,
#             owl_predictor: OwlPredictor
#         ):
#         self.clip_predictor = clip_predictor
#         self.owl_predictor = owl_predictor

#     def encode_text(self, text: List[str]):
#         pass

#     def predict(self, image: PIL.Image, tree: Tree):
#         pass