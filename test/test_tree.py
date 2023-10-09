import pytest
from nanoowl.tree import (
    Tree,
    TreeOp
)


def test_tree_from_prompt():

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


def test_tree_to_dict():
    
    tree = Tree.from_prompt("[a[b,c(d,e)]]")
    tree_dict = tree.to_dict()
    assert "nodes" in tree_dict
    assert "labels" in tree_dict
    assert len(tree_dict["nodes"]) == 3
    assert len(tree_dict["labels"]) == 6



def test_tree_from_prompt():

    tree = Tree.from_prompt("(office, home, outdoors, gym)")

    print(tree)