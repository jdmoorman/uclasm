"""TODO: Docstring."""
import pytest
import uclasm
from uclasm import Graph
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd


@pytest.fixture
def graph():
    """Create a graph with some nodes and edges."""
    adj0 = csr_matrix([[0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0]])
    adj1 = csr_matrix([[0, 0, 0],
                       [0, 0, 0],
                       [0, 1, 0]])
    nodelist = pd.DataFrame(['a', 'b', 'c'], columns=[Graph.node_col])
    edgelist = pd.DataFrame([['b', 'a'],
                             ['c', 'b']], columns=[Graph.source_col,
                                                   Graph.target_col])
    g = Graph([adj0, adj1], ['c1', 'c2'], nodelist, edgelist)
    return g


@pytest.fixture
def node_subgraph():
    """Create a subgraph of the graph created by the graph fixture."""
    adj0 = csr_matrix([[0, 0],
                       [0, 0]])
    adj1 = csr_matrix([[0, 0],
                       [1, 0]])
    nodelist = pd.DataFrame(['b', 'c'], columns=[Graph.node_col])
    edgelist = pd.DataFrame([['c', 'b']], columns=[Graph.source_col,
                                                   Graph.target_col])
    subg = Graph([adj0, adj1], ['c1', 'c2'], nodelist, edgelist)
    return subg


class TestGraph:
    """Tests related to the Graph class and its methods/functions."""

    def test_graph_adj_mats(self, graph):
        """Check that the various derived matrices are computed."""
        adj0 = graph.adjs[0]
        adj1 = graph.adjs[1]

        comp = adj0 + adj1
        sym_comp = comp + comp.T
        is_nbr = sym_comp > 0

        # Check the various composite adjacency and related matrices.
        assert (graph.composite_adj != comp).nnz == 0
        assert (graph.sym_composite_adj != sym_comp).nnz == 0
        assert (graph.is_nbr != is_nbr).nnz == 0

    def test_node_subgraph(self, graph, node_subgraph):
        """Check the subgraph of `graph` against the expected subgraph."""
        node_bools = graph.nodelist[Graph.node_col]\
                     .isin(node_subgraph.nodelist[Graph.node_col])
        node_idxs = node_bools[node_bools].index
        actual_subgraph = graph.node_subgraph(node_idxs)

        # Check the adjacency matrices of the subgraph against expected
        for act_adj, exp_adj in zip(actual_subgraph.adjs, node_subgraph.adjs):
            assert (act_adj != exp_adj).nnz == 0
            assert (act_adj != exp_adj).nnz == 0
        assert (actual_subgraph.nodelist == node_subgraph.nodelist).all(axis=None)
        assert (actual_subgraph.edgelist == node_subgraph.edgelist).all(axis=None)

    def test_node_cover(self, graph):
        """Check that the node cover is minimal for the example graph."""
        cover = graph.node_cover()
        assert len(cover) == 1
        assert cover[0] == 1

costs_list = []

np.random.seed(0)
costs = np.random.normal(size=(4, 10))
costs -= np.min(costs)
costs[0, 0] = 0
costs[1, 0] = 0.2
costs_list.append(costs)

costs = np.array([[0, 2, 2],
                  [0, 1, 2],
                  [0, 0, 1]])
costs_list.append(costs)

@pytest.mark.parametrize("costs", costs_list)
def test_constrained_lsap(costs):
    """TODO: Docstring."""
    costs = np.random.normal(size=(4, 10))
    costs -= np.min(costs)
    costs[0, 0] = 0
    costs[1, 0] = 0.2
    total_costs = uclasm.constrained_lsap_costs(costs)
    for i, j in np.ndindex(*costs.shape):
        actual_total_cost = uclasm.constrained_lsap_cost(i, j, costs)
        assert pytest.approx(actual_total_cost) == total_costs[i, j]
