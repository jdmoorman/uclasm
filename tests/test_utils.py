import pytest
import uclasm
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

@pytest.fixture
def graph():
    adj0 = csr_matrix([[0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0]])
    adj1 = csr_matrix([[0, 0, 0],
                       [0, 0, 0],
                       [0, 1, 0]])
    nodelist = pd.DataFrame(['a', 'b', 'c'], columns=['node'])
    edgelist = pd.DataFrame([['b', 'a'], ['c', 'b']], columns=['src', 'dest'])
    g = uclasm.Graph([adj0, adj1], ['c1', 'c2'], nodelist, edgelist)
    return g

@pytest.fixture
def subgraph():
    adj0 = csr_matrix([[0, 0],
                       [0, 0]])
    adj1 = csr_matrix([[0, 0],
                       [1, 0]])
    nodelist = pd.DataFrame(['b', 'c'], columns=['node'])
    edgelist = pd.DataFrame([['c', 'b']], columns=['src', 'dest'])
    subg = uclasm.Graph([adj0, adj1], ['c1', 'c2'], nodelist, edgelist)
    return subg


class TestGraph:
    def test_graph_adj_mats(self, graph):
        adj0 = graph.adjs[0]
        adj1 = graph.adjs[1]

        comp = adj0 + adj1
        sym_comp = comp + comp.T
        is_nbr = sym_comp > 0

        # Check the various composite adjacency and related matrices.
        assert (graph.composite_adj != comp).nnz == 0
        assert (graph.sym_composite_adj != sym_comp).nnz == 0
        assert (graph.is_nbr != is_nbr).nnz == 0


    def test_subgraph(self, graph, subgraph):
        """Check the subgraph of `graph` against the expected subgraph"""
        node_bools = graph.nodelist['node'].isin(subgraph.nodelist['node'])
        node_idxs = node_bools[node_bools].index
        actual_subgraph = graph.subgraph(node_idxs)

        # Check the adjacency matrices of the subgraph against expected
        for act_adj, exp_adj in zip(actual_subgraph.adjs, subgraph.adjs):
            assert (act_adj != exp_adj).nnz == 0
            assert (act_adj != exp_adj).nnz == 0
        assert (actual_subgraph.nodelist == subgraph.nodelist).all(axis=None)
        assert (actual_subgraph.edgelist == subgraph.edgelist).all(axis=None)

    def test_node_cover(self, graph):
        cover = graph.node_cover()
        assert len(cover) == 1
        assert cover[0] == 1

