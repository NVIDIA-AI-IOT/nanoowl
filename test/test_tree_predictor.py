import pytest
from nanoowl.tree_predictor import (
    TreeNode,
    TreeOpType,
    TreeOp
)


def test_tree_node_branch_parse_child_nodes():
    pass


def test_tree_node_branch_parse_label_text():
    assert TreeNode.parse_label_text("a") == "a"
    assert TreeNode.parse_label_text("a(b,c)") == "a"
    assert TreeNode.parse_label_text("a[b,c]") == "a"
    assert TreeNode.parse_label_text("a[b(d,e),c]") == "a"
    assert TreeNode.parse_label_text("a(b,c)") == "a"# throw error?


def test_tree_node_branch_parse_output_branches():
    pass