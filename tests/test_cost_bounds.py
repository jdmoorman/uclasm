"""Tests for the cost bounds."""
import pytest
import uclasm
from uclasm import Graph, MatchingProblem
from uclasm.matching import *
from uclasm.matching import *
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
def smp_nbhd_noisy():
    """Create a subgraph matching problem."""
    adj_tmplt = csr_matrix([[0, 0, 1, 0],
                            [0, 0, 1, 0],
                            [1, 1, 0, 1],
                            [0, 0, 0, 0]])
    nodelist = pd.DataFrame(['a', 'b', 'c', 'd'], columns=[Graph.node_col])
    edgelist = pd.DataFrame([['a', 'c', 'c1'],
                             ['c', 'a', 'c1'],
                             ['b', 'c', 'c1'],
                             ['c', 'b', 'c1'],
                             ['c', 'd', 'c1']], columns=[Graph.source_col,
                                                   Graph.target_col,
                                                   Graph.channel_col])
    tmplt = Graph([adj_tmplt], ['c1'], nodelist, edgelist)
    adj_world = csr_matrix([[0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 2, 1, 0],
                            [0, 1, 2, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]])
    nodelist = pd.DataFrame(['1', '2', '3', '4', '5', '6'], columns=[Graph.node_col])
    edgelist = pd.DataFrame([['1', '3', 'c1'],
                             ['3', '1', 'c1'],
                             ['3', '5', 'c1'],
                             ['3', '4', 'c1'],
                             ['3', '4', 'c1'],
                             ['4', '3', 'c1'],
                             ['4', '3', 'c1'],
                             ['4', '2', 'c1'],
                             ['4', '6', 'c1'],], columns=[Graph.source_col,
                                                   Graph.target_col,
                                                   Graph.channel_col])
    world = Graph([adj_world], ['c1'], nodelist, edgelist)
    smp = MatchingProblem(tmplt, world)
    return smp

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

class TestNeighborhoodCostBound:
    def test_neighborhood_cost_noisy(self, smp_nbhd_noisy):
        local_cost_bound.neighborhood(smp_nbhd_noisy)
        # Note: in nbhd cost bound, the tmplt nodes with only one neighbor are
        # passed since this is equiv to edgewise cost bound
        # TODO: write a comprehensive test for all cost bounds 
        final_cost = [[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [3, 4, 0, 1, 4, 4],
                      [0, 0, 0, 0, 0, 0]]
        print(smp_nbhd_noisy.local_costs)
        for i in range(4):
            for j in range(6):
                assert(final_cost[i][j] == smp_nbhd_noisy.local_costs[i][j])
