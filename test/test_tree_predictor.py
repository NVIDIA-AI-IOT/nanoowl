import pytest
from nanoowl.tree_predictor import (
    TreeNode,
    TreeNodeType,
    TreeBranch
)


def test_tree_node_from_prompt():

    root_node = TreeNode.from_prompt("[a face]")

    assert root_node.type == TreeNodeType.INPUT
    assert root_node.branches[0].label == ""
    assert root_node.branches[0].parent == root_node

    detect_node = root_node.branches[0].nodes[0]

    assert detect_node.branches[0].label == "a face"
    assert detect_node.branches[0].parent == detect_node