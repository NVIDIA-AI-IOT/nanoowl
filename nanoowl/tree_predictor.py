import json
from enum import Enum
from typing import List, Mapping, Optional


__all__ = [
    "TreeNodeType",
    "TreeBranch",
    "TreeNode",

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