"""Tests for the cost bounds."""
import pytest
import uclasm
from uclasm import Graph, MatchingProblem
from uclasm.matching.cost_bounds import *
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
    smp = MatchingProblem(tmplt, world, cost_threshold=1)
    return smp

class TestEdgewiseCostBound:
    """Tests related to the edgewise cost bound """
    def test_edgewise_cost(self, smp):
        edgewise_cost_bound(smp)
        assert(np.sum(smp.candidates()) == 3)

    def test_edgewise_cost_noisy(self, smp_noisy):
        edgewise_cost_bound(smp_noisy)
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
                assert(final_cost[i][j] == smp_noisy.structural_costs[i][j])
        assert(np.sum(smp_noisy.candidates()) == 3)
