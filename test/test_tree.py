# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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