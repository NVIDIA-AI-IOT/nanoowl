from .tree import Tree
from .owl_predictor import OwlPredictor, OwlEncodeTextOutput
from .clip_predictor import ClipPredictor, ClipEncodeTextOutput
from .image_preprocessor import ImagePreprocessor

import torch
import PIL.Image
from typing import Optional, Tuple, List, Mapping
from dataclasses import dataclass


@dataclass
class TreeRoi:
    id: int
    parent_id: int
    box: Tuple[float, float, float, float]
    labels: List[int]


class TreePredictor(torch.nn.Module):

    def __init__(self,
            owl_predictor: Optional[OwlPredictor] = None,
            clip_predictor: Optional[ClipPredictor] = None,
            image_preprocessor: Optional[ImagePreprocessor] = None
        ):
        super().__init__()
        self.owl_predictor = OwlPredictor() if owl_predictor is None else owl_predictor
        self.clip_predictor = ClipPredictor() if clip_predictor is None else clip_predictor
        self.image_preprocessor = ImagePreprocessor() if image_preprocessor is None else image_preprocessor

    def encode_clip_labels(self, tree: Tree) -> Mapping[int, ClipEncodeTextOutput]:
        label_indices = tree.get_classify_label_indices()
        labels = [tree.labels[index] for index in label_indices]
        text_encodings = self.clip_predictor.encode_text(labels)
        label_encodings = {}
        for i in range(len(labels)):
            label_encodings[label_indices[i]] = text_encodings.slice(i, i+1)
        return label_encodings
    
    def encode_owl_labels(self, tree: Tree) -> Mapping[int, OwlEncodeTextOutput]:
        label_indices = tree.get_detect_label_indices()
        labels = [tree.labels[index] for index in label_indices]
        text_encodings = self.owl_predictor.encode_text(labels)
        label_encodings = {}
        for i in range(len(labels)):
            label_encodings[label_indices[i]] = text_encodings.slice(i, i+1)
        return label_encodings
    
    def predict(self, image: PIL.Image.Image, tree: Tree):
        
        image_tensor = self.image_preprocessor(image)

        root_roi = TreeRoi(
            id=0,
            parent_id=-1,
            box=(0, 0, image.width, image.height),
            labels=[0]
        )

        rois = [root_roi]

        return rois