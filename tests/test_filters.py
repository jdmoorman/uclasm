"""Tests for the search functions."""
import pytest
import uclasm
from uclasm import Graph, MatchingProblem
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
    smp = MatchingProblem(tmplt, world, global_cost_threshold=1,
                          local_cost_threshold=1)
    return smp

class TestFilters:
    """Tests related to the filters """
    def test_stats_filter(self, smp):
        filters.stats_filter(smp)
        assert np.sum(smp.local_costs > 0) == 6

    def test_stats_filter_noisy(self, smp_noisy):
        filters.stats_filter(smp_noisy)
        assert np.sum(smp_noisy.local_costs > 0) == 2

    def test_topology_filter(self, smp):
        filters.topology_filter(smp)
        assert np.sum(smp.local_costs > 0) == 6

    def test_topology_filter_noisy(self, smp_noisy):
        filters.topology_filter(smp_noisy)
        assert np.sum(smp_noisy.local_costs > 0) == 2

    def test_run_filters(self, smp):
        filters.run_filters(smp)
        assert np.sum(smp.candidates()) == 3

    def test_run_filters_noisy(self, smp_noisy):
        filters.run_filters(smp_noisy)
        assert np.sum(smp_noisy.candidates()) == 5
