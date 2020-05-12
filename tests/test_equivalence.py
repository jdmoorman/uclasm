"""
Tests for structural equivalence.
"""
import pytest
import networkx as nx

from uclasm.graph import from_networkx_graph


@pytest.fixture
def star_graph_5():
    """
    Creates the star graph S5, the graph with 1 central node and 5 spokes.
    The node with index 0 should be the central node.
    """
    star_graph = nx.star_graph(5)
    return from_networkx_graph(star_graph)


@pytest.fixture
def complete_graph_5():
    """
    Creates the complete graph K5, the graph with 5 nodes each connected to
    every other node.
    """
    complete_graph = nx.complete_graph(5)
    return from_networkx_graph(complete_graph)


@pytest.fixture
def path_graph_5():
    """
    Creates the path graph P5, the path graph with 5 nodes.
    """
    path_graph = nx.path_graph(5)
    return from_networkx_graph(path_graph)


def test_star_graph_equivalence(star_graph_5):
    equiv = star_graph_5.equivalence_classes
    assert len(equiv.classes) == 2
    assert equiv.classes[0] == {0}
    rep_1 = equiv.representative(1)
    assert equiv.classes[rep_1] == {1,2,3,4,5}


def test_complete_graph_equivalence(complete_graph_5):
    equiv = complete_graph_5.equivalence_classes
    assert len(equiv.classes) == 1
    rep = equiv.representative(0)
    assert equiv.classes[rep] == {0,1,2,3,4}


def test_path_graph_equivalence(path_graph_5):
    equiv = path_graph_5.equivalence_classes
    assert len(equiv.classes) == 5
    for i in range(5):
        assert equiv.classes[i] == {i}
