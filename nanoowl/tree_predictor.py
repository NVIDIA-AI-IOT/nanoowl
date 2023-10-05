import torch
import PIL.Image
from typing import List
from dataclasses import dataclass
from .clip_predictor import ClipPredictor
from .owl_predictor import OwlPredictor


__all__ = [
    "TreePredictor"
]


@dataclass
class Tree:
    pass


class TreePredictor(object):
    def __init__(self,
            clip_predictor: ClipPredictor,
            owl_predictor: OwlPredictor
        ):
        self.clip_predictor = clip_predictor
        self.owl_predictor = owl_predictor

    def encode_text(self, text: List[str]):
        pass

    def predict(self, image: PIL.Image, tree: Tree):
        pass