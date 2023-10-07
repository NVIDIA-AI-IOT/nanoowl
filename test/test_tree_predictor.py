import pytest
from nanoowl.tree_predictor import (
    Tree,
    TreeOp
)


def test_tree_node_from_prompt():

    graph = Tree.from_prompt("[a face]")

    assert len(graph.nodes) == 1
    assert len(graph.labels) == 2
    assert graph.labels[0] == "image"
    assert graph.labels[1] == "a face"
    assert graph.nodes[0].op == TreeOp.DETECT

    graph = Tree.from_prompt("[a face](a dog, a cat)")

    assert len(graph.nodes) == 2
    assert len(graph.labels) == 4
    assert graph.labels[0] == "image"
    assert graph.labels[1] == "a face"
    assert graph.labels[2] == "a dog"
    assert graph.labels[3] == "a cat"
    assert graph.nodes[0].op == TreeOp.DETECT
    assert graph.nodes[1].op == TreeOp.CLASSIFY

    with pytest.raises(RuntimeError):
        Tree.from_prompt("]a face]")

    with pytest.raises(RuntimeError):
        Tree.from_prompt("[a face")

    with pytest.raises(RuntimeError):
        Tree.from_prompt("[a face)")

    with pytest.raises(RuntimeError):
        Tree.from_prompt("[a face]]")