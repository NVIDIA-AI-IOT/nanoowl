from enum import Enum
from typing import List, Mapping


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

    def __init__(self, op: TreeOp, input: int):
        self.op = op
        self.input = input
        self.outputs = []


class Tree:
    nodes: List[TreeNode]
    labels: Mapping[int, str]

    def __init__(self, nodes, labels):
        self.nodes = nodes
        self.labels = labels

    @staticmethod
    def from_prompt(prompt: str):

        nodes = []
        node_stack = []
        label_index_stack = [0]
        labels = {0: "image"}
        label_index = 0

        for ch in prompt:

            if ch == "[":
                label_index += 1
                node = TreeNode(op=TreeOp.DETECT, input=label_index_stack[-1])
                node.outputs.append(label_index)
                node_stack.append(node)
                label_index_stack.append(label_index)
                labels[label_index] = ""
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
                labels[label_index] = ""
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
                labels[label_index] = ""
            else:
                labels[label_index_stack[-1]] += ch

        if len(node_stack) > 0:
            if node_stack[-1].op == TreeOp.DETECT:
                raise RuntimeError("Missing ']'.")
            if node_stack[-1].op == TreeOp.CLASSIFY:
                raise RuntimeError("Missing ')'.")
            
        labels = {k: v.strip() for k, v in labels.items()}

        graph = Tree(nodes=nodes, labels=labels)

        return graph