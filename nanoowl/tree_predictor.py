import torch
import PIL.Image
from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from .clip_predictor import ClipPredictor
from .owl_predictor import OwlPredictor


__all__ = [
    "TreeNodeType",
    "TreeBranch",
    "TreeNode",

]


class TreeNodeType(Enum):
    INPUT = "input"
    DETECT = "detect"
    CLASSIFY = "classify"

    def __str__(self):
        return self.name

class TreeBranch:
    label: str
    parent: "TreeNode"
    nodes: List["TreeNode"]
    global_id: int

    def __init__(self, parent: "TreeNode", id: int, label: str = ""):
        self.parent = parent
        self.label = label
        self.nodes = []
        self.global_id = id

    def add_node(self, type: TreeNodeType):
        node = TreeNode(type=type, input=self)
        self.nodes.append(node)
        return node
    
    def extend_label(self, ch: str):
        self.label += ch

    def as_dict(self):
        return {
            "label": self.label,
            "nodes": [node.as_dict() for node in self.nodes],
            "global_id": self.global_id
        }
    

class TreeNode:
    type: TreeNodeType
    input: Optional[TreeBranch]
    branches: List[TreeBranch]

    def __init__(self, type: TreeNodeType, input: Optional[TreeBranch] = None):
        self.type = type
        self.input = input
        self.branches = []

    def add_branch(self, id: int, label: str = "") -> TreeBranch:
        branch = TreeBranch(self, id, label)
        self.branches.append(branch)
        return branch
    
    @staticmethod
    def from_prompt(prompt: str):
        
        global_branch_id = 0
        branch = TreeNode(TreeNodeType.INPUT).add_branch(global_branch_id, "input")

        for ch in prompt:

            if ch == "[":
                global_branch_id += 1
                branch = branch.add_node(TreeNodeType.DETECT).add_branch(global_branch_id)
            elif ch == "]":
                if branch.parent.type != TreeNodeType.DETECT:
                    raise RuntimeError("Unexpected ']'.")
                if branch.parent.input is None:
                    raise RuntimeError("Unexpected ']'.")
                branch = branch.parent.input
            elif ch == "(":
                global_branch_id += 1
                branch = branch.add_node(TreeNodeType.CLASSIFY).add_branch(global_branch_id)
            elif ch == ")":
                if branch.parent.type != TreeNodeType.CLASSIFY:
                    raise RuntimeError("Unexpected ')'.")
                if branch.parent.input is None:
                    raise RuntimeError("Unexpected ')'.")
                branch = branch.parent.input
            elif ch == ",":
                global_branch_id += 1
                branch = branch.parent.add_branch(global_branch_id)
            else:
                branch.extend_label(ch)

        return branch.parent
    
    def as_dict(self):
        return {
            "type": str(self.type),
            "branches": [branch.as_dict() for branch in self.branches]
        }
