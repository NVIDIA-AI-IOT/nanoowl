from typing import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TreeOp(Enum):
    ROOT = "ROOT"
    DETECT = "DETECT"
    CLASSIFY = "CLASSIFY"


@dataclass
class _ParseTreeNode:
    op: TreeOp
    labels: List[str]
    children: List["_ParseTreeNode"] = field(default_factory=lambda: [])
    label_ids: Optional[List[int]] = None
    node_id: Optional[int] = None
    parent: Optional["_ParseTreeNode"] = None
    parent_label_index: Optional[int] = None

    @staticmethod
    def from_prompt(prompt: str):
        return _parse_tree_prompt(prompt)


def _assign_label_ids(root: _ParseTreeNode):

    queue = [root]
    current_id = 0
    
    while queue:
        node = queue.pop(0)
        node.label_ids = []
        for label in node.labels:
            node.label_ids.append(current_id)
            current_id += 1
        queue += node.children


def _assign_node_ids(root: _ParseTreeNode):

    queue = [root]
    current_id = 0
    
    while queue:
        node = queue.pop(0)
        node.node_id = current_id
        current_id += 1
        queue += node.children


def _parse_tree_prompt(prompt: str):
    
    stack = [_ParseTreeNode(op=TreeOp.ROOT, labels=[""], parent=None)]

    for ch in prompt:
        if ch == "[":
            parent = stack[-1]
            child = _ParseTreeNode(op=TreeOp.DETECT, labels=[""], parent=parent, parent_label_index=len(parent.labels) - 1)
            parent.children.append(child)
            stack.append(child)
        elif ch == "(":
            parent = stack[-1]
            child = _ParseTreeNode(op=TreeOp.CLASSIFY, labels=[""], parent=parent, parent_label_index=len(parent.labels) - 1)
            parent.children.append(child)
            stack.append(child)
        elif ch == ",":
            stack[-1].labels.append("")
        elif ch == "]":
            if stack[-1].op != TreeOp.DETECT:
                raise RuntimeError("Prompt error.  Unexpected ].")
            stack.pop()
        elif ch == ")":
            if stack[-1].op != TreeOp.CLASSIFY:
                raise RuntimeError("Prompt error.  Unexpected ).")
            stack.pop()
        else:
            stack[-1].labels[-1] += ch
    
    if len(stack) > 1:

        if stack[-1].op == TreeOp.DETECT:
            raise RuntimeError("Prompt error.  Missing ].")

        if stack[-1].op == TreeOp.CLASSIFY:
            raise RuntimeError("Prompt error.  Missing ).")

    root = stack[-1]
    _assign_label_ids(root)
    _assign_node_ids(root)

    return root


@dataclass
class TreeNode:
    id: int
    op: TreeOp
    labels: List[str]
    input_buffer: int
    output_buffers: List[int]


@dataclass
class Tree:
    nodes: List[TreeNode]

    @staticmethod
    def from_tree(root: _ParseTreeNode):
        _assign_label_ids(root)
        _assign_node_ids(root)

        queue = root.children
        tree_nodes = []
        num_nodes = 0
        while queue:
            parse_tree = queue.pop(0)
            tree_node = TreeNode(
                id=num_nodes,
                op=parse_tree.op,
                labels=[l.strip() for l in parse_tree.labels],
                input_buffer=parse_tree.parent.label_ids[parse_tree.parent_label_index],
                output_buffers=parse_tree.label_ids
            )
            num_nodes += 1
            tree_nodes.append(tree_node)
            queue += parse_tree.children

        return Tree(nodes=tree_nodes)

    @staticmethod
    def from_prompt(prompt: str):
        return Tree.from_tree(_ParseTreeNode.from_prompt(prompt))

    def print(self):
        for n in self.nodes:
            print(n)

    def detect_nodes(self):
        return [n for n in self.nodes if n.op == TreeOp.DETECT]

    def classify_nodes(self):
        return [n for n in self.nodes if n.op == TreeOp.CLASSIFY]

    def get_node_for_output_buffer(self, output_buffer: int):
        return next(n for n in self.nodes if output_buffer in n.output_buffers)

    def get_nodes_with_input_buffer(self, input_buffer: int):
        return [n for n in self.nodes if input_buffer == n.input_buffer]

    def find_nodes(self,
            input_buffer: Optional[int] = None,
            output_buffer: Optional[int] = None,
            op: Optional[TreeOp] = None
        ):

        nodes = []
        for n in self.nodes:
            if input_buffer is not None and n.input_buffer != input_buffer:
                continue
            if output_buffer is not None and output_buffer not in n.output_buffers:
                continue
            if op is not None and n.op != op:
                continue
            nodes.append(n)
        return nodes

    def get_all_buffers(self):
        buffers = [0] # TODO: assumes 0 is used as input for at least one node
        for n in self.nodes:
            buffers += n.output_buffers
        return buffers

    def get_leaf_buffers(self):
        buffers = self.get_all_buffers()
        return [b for b in buffers if len(self.find_nodes(input_buffer=b)) == 0]
    
    def get_buffer_label(self, buffer: int):
        nodes = self.find_nodes(output_buffer=buffer)
        if len(nodes) == 0:
            return None
        node = nodes[0]
        index = node.output_buffers.index(buffer)
        return node.id, index, node.labels[index]

    def get_buffer_label_map(self):
        all_buffers = self.get_all_buffers()
        label_map = {}
        for buf in all_buffers:
            label_map[buf] = self.get_buffer_label(buf)
        return label_map

    def get_node_with_id(self, node_id: int):
        return next(n for n in self.nodes if n.id == node_id)

    def get_buffer_depth(self, buffer):
        nodes = self.find_nodes(output_buffer=buffer)
        if len(nodes) == 0:
            return -1
        node = nodes[0]
        depth = 0
        while node.input_buffer != 0:
            depth += 1
            nodes = self.find_nodes(output_buffer=node.input_buffer)
            if len(nodes) == 0:
                break
            node = nodes[0]
        return depth

    def get_buffer_depth_map(self):
        all_buffers = self.get_all_buffers()
        depth_map = {}
        for buf in all_buffers:
            depth_map[buf] = self.get_buffer_depth(buf)
        return depth_map