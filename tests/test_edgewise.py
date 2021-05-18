"""Tests for the cost bounds."""
import pytest
import uclasm
from uclasm import Graph, MatchingProblem
from uclasm.matching import *
from uclasm.matching import *
from uclasm.matching.local_cost_bound.edgewise import *
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd


@pytest.fixture
def smp():
    """Create a subgraph matching problem."""
    adj0 = csr_matrix([[0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0]])
    adj1 = csr_matrix([[0, 0, 0],
                       [0, 0, 0],
                       [0, 1, 0]])
    nodelist = pd.DataFrame(['a', 'b', 'c'], columns=[Graph.node_col])
    edgelist = pd.DataFrame([['b', 'a', 'c1'],
                             ['c', 'b', 'c2']], columns=[Graph.source_col,
                                                   Graph.target_col,
                                                   Graph.channel_col])
    tmplt = Graph([adj0, adj1], ['c1', 'c2'], nodelist, edgelist)
    world = Graph([adj0, adj1], ['c1', 'c2'], nodelist, edgelist)
    smp = MatchingProblem(tmplt, world)
    return smp

@pytest.fixture
def smp_noisy():
    """Create a noisy subgraph matching problem."""
    adj0 = csr_matrix([[0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0]])
    adj1 = csr_matrix([[0, 0, 0],
                       [0, 0, 0],
                       [0, 1, 0]])
    nodelist = pd.DataFrame(['a', 'b', 'c'], columns=[Graph.node_col])
    edgelist = pd.DataFrame([['b', 'a', 'c1'],
                             ['c', 'b', 'c2']], columns=[Graph.source_col,
                                                   Graph.target_col,
                                                   Graph.channel_col])
    tmplt = Graph([adj0, adj1], ['c1', 'c2'], nodelist, edgelist)
    adj2 = csr_matrix(np.zeros((3,3)))
    edgelist2 = pd.DataFrame([['b', 'a', 'c1']], columns=[Graph.source_col,
                                                   Graph.target_col,
                                                   Graph.channel_col])
    world = Graph([adj0.copy(), adj2], ['c1', 'c2'], nodelist, edgelist2)
    smp = MatchingProblem(tmplt, world, global_cost_threshold=1)
    return smp

@pytest.fixture
def smp_noisy_bidirectional():
    """Create a noisy subgraph matching problem."""
    adj0 = csr_matrix([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])
    tmplt = Graph([adj0])
    zero_adj = csr_matrix(np.zeros((3,3)))
    world = Graph([zero_adj])
    smp = MatchingProblem(tmplt, world, global_cost_threshold=2)
    return smp

@pytest.fixture
def edgelist():
    """Create a pandas DataFrame representing an edgelist"""
    edgelist = pd.DataFrame([['b', 'a', '1', '2', '3'],
                             ['c', 'b', '2', '3', '1'],
                             ['b', 'c', '2', '3', '1'],
                             ['c', 'b', '1', '2', '1'],
                             ['c', 'b', '1', '2', '3'],
                            ],
                            columns=['source_col', 'target_col',
                                     'attr1', 'attr2', 'attr3'])
    return edgelist

class TestEdgewiseCostBound:
    """Tests related to the edgewise cost bound """
    def test_edgewise_cost(self, smp):
        local_cost_bound.edgewise(smp)
        global_cost_bound.from_local_bounds(smp)
        assert(np.sum(smp.candidates()) == 3)

    def test_edgewise_cost_noisy(self, smp_noisy):
        local_cost_bound.edgewise(smp_noisy)
        global_cost_bound.from_local_bounds(smp_noisy)
        # First edge: increases cost for any match of a, b that isn't a:a, b:b
        # cost: [0 1 1]
        #       [1 0 1]
        #       [0 0 0]
        # Second edge: increases cost for all matches of b,c
        # cost: [0 1 1]
        #       [2 1 2]
        #       [1 1 1]
        final_cost = [[0, 1, 1],
                      [2, 1, 2],
                      [1, 1, 1]]
        for i in range(3):
            for j in range(3):
                assert(final_cost[i][j] == smp_noisy.local_costs[i][j])
        assert(np.sum(smp_noisy.candidates()) == 3)

    def test_edgewise_cost_noisy(self, smp_noisy_bidirectional):
        local_cost_bound.edgewise(smp_noisy_bidirectional)
        global_cost_bound.from_local_bounds(smp_noisy_bidirectional)
        # First edge: increases cost for any match of a, b that isn't a:a, b:b
        # cost: [0 1 1]
        #       [1 0 1]
        #       [0 0 0]
        # Second edge: increases cost for all matches of b,c
        # cost: [0 1 1]
        #       [2 1 2]
        #       [1 1 1]
        final_cost = [[2, 2, 2],
                      [2, 2, 2],
                      [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                assert(final_cost[i][j] == smp_noisy_bidirectional.local_costs[i][j])
        assert(np.sum(smp_noisy_bidirectional.candidates()) == 9)

    def test_edge_attr_fn(self, smp):
        smp.edge_attr_fn = lambda edge_key, sub_edge_key, tmplt_attrs_dict, cand_attrs_dict, importance_value: 0 if tmplt_attrs_dict[smp.tmplt.channel_col] == cand_attrs_dict[smp.world.channel_col] else 1
        smp.missing_edge_cost_fn = lambda tmplt_row: 1000
        local_cost_bound.edgewise(smp)
        global_cost_bound.from_local_bounds(smp)
        assert(np.sum(smp.candidates()) == 3)

    def test_get_edge_to_unique_attr(self, edgelist):
        unique_attrs, inverse = get_edge_to_unique_attr(edgelist, 'source_col', 'target_col')
        assert(len(unique_attrs) == 3)
        for idx, row in edgelist.iterrows():
            for idx2, attr in enumerate(['attr1','attr2','attr3']):
                print(type(inverse[idx]))
                print(unique_attrs)
                assert unique_attrs[inverse[idx], idx2] == row[attr]

    def test_get_edge_to_unique_attr_str_cast(self, edgelist):
        unique_attrs, inverse = get_edge_to_unique_attr(edgelist, 'source_col', 'target_col', cast_to_str=True)
        assert(len(unique_attrs) == 3)
        for idx, row in edgelist.iterrows():
            for idx2, attr in enumerate(['attr1','attr2','attr3']):
                print(type(inverse[idx]))
                print(unique_attrs)
                assert unique_attrs[inverse[idx], idx2] == row[attr]
