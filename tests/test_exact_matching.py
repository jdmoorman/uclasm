"""Tests for the search functions."""
import pytest
import uclasm
from uclasm import Graph, ExactMatchingProblem
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
    smp = ExactMatchingProblem(tmplt, world)
    return smp


class TestMatching:
    """Tests related to the matching"""
    def test_add_match(self, smp):
        assert len(smp.matching) == 0
        smp.add_match(1, 1)
        assert len(smp.matching) == 1
        assert smp.candidates[1,1]
        assert np.sum(smp.candidates[1,:]) == 1
        assert np.sum(smp.candidates[:,1]) == 1

    def test_enforce_matching(self, smp):
        smp.enforce_matching(((0,1),(1,2)))
        assert len(smp.matching) == 2
        assert smp.candidates[0,1]
        assert smp.candidates[1,2]
        assert np.sum(smp.candidates[:2,:]) == 2
        assert np.sum(smp.candidates[:,1:]) == 2

    def test_prevent_match(self, smp):
        smp.prevent_match(1,2)
        assert not smp.candidates[1,2]

class TestFilters:
    """Tests related to the filters """
    def test_stats_filter(self, smp):
        filters.stats_filter(smp)
        assert np.sum(smp.candidates) == 3

    def test_topology_filter(self, smp):
        filters.topology_filter(smp)
        assert np.sum(smp.candidates) == 3

    def test_permutation_filter(self, smp):
        filters.permutation_filter(smp)
        assert np.sum(smp.candidates) == 9
        smp.candidates[0, 0] = False
        smp.candidates[1, 0] = False
        filters.permutation_filter(smp)
        assert np.sum(smp.candidates) == 5
        assert smp.candidates[2, 0]
        assert np.sum(smp.candidates[2,:]) == 1

    def test_gac_filter(self, smp):
        filters.gac_filter(smp)
        assert np.sum(smp.candidates) == 9
        smp.candidates[0, 0] = False
        smp.candidates[1, 0] = False
        filters.gac_filter(smp)
        assert np.sum(smp.candidates) == 5
        assert smp.candidates[2, 0]
        assert np.sum(smp.candidates[2,:]) == 1

    def test_elimination_filter(self, smp):
        filters.elimination_filter(smp)
        assert np.sum(smp.candidates) == 3

    def test_neighborhood_filter(self, smp):
        filters.neighborhood_filter(smp)
        # Neighborhood filter skips nodes with one edge since those are covered
        # by topology filter, so only the center node is checked here
        assert np.sum(smp.candidates) == 7
        smp.candidates[:, :] = True
        # Disqualify the only candidate for the center node by removing the node
        # connected to it
        smp.candidates[0, 0] = False
        filters.neighborhood_filter(smp)
        assert np.sum(smp.candidates[1, :]) == 0

    def test_validation_filter(self, smp):
        filters.validation_filter(smp)
        assert np.sum(smp.candidates) == 3

    def test_run_filters(self, smp):
        filters.run_filters(smp)
        assert np.sum(smp.candidates) == 3
