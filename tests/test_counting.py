"""Tests for the counting."""
import pytest
import uclasm
from uclasm.counting import count_alldiffs, count_isomorphisms
from uclasm.matching.search.search_utils import iterate_to_convergence
from uclasm import Graph, MatchingProblem

import pandas as pd
from scipy.sparse import csr_matrix

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
def smp_star():
    """Create a more complicated subgraph matching problem."""
    # degree-4 node surrounded by 2 pairs of 2 nodes in each channel
    adj0 = csr_matrix([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
    adj1 = csr_matrix([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
    nodelist = pd.DataFrame(['a', 'b', 'c', 'd', 'e'], columns=[Graph.node_col])
    edgelist = pd.DataFrame([['c', 'a', 'c1'],
                             ['c', 'b', 'c1'],
                             ['c', 'd', 'c2'],
                             ['c', 'e', 'c2']], columns=[Graph.source_col,
                                                   Graph.target_col,
                                                   Graph.channel_col])
    tmplt = Graph([adj0, adj1], ['c1', 'c2'], nodelist, edgelist)
    world = Graph([adj0, adj1], ['c1', 'c2'], nodelist, edgelist)
    smp = MatchingProblem(tmplt, world)
    return smp

class TestAlldiffs:
    def test_disjoint_sets(self):
        """Number of solutions should be product of candidate set sizes for disjoint
        candidate sets."""

        d = {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7]
        }
        assert count_alldiffs(d) == 12

        d = {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        }
        assert count_alldiffs(d) == 8

    def test_simple_cases(self):
        d = {
            "a": [1, 2],
            "b": [2],
            "c": [3, 4],
        }
        assert count_alldiffs(d) == 2
        d = {
            "a": [1, 2],
            "b": [2, 3],
            "c": [3, 1],
        }
        assert count_alldiffs(d) == 2
        d = {
            "a": [1, 2],
            "b": [1, 2, 3],
            "c": [3, 1],
        }
        assert count_alldiffs(d) == 3
        d = {
            "a": [1, 2],
            "b": [],
            "c": [3, 4],
        }
        assert count_alldiffs(d) == 0

class TestIsomorphisms:
    def test_count_isomorphisms(self, smp):
        iterate_to_convergence(smp)
        count = count_isomorphisms(smp, verbose=True)
        assert count == 1

    def test_count_isomorphisms(self, smp_star):
        iterate_to_convergence(smp_star)
        count = count_isomorphisms(smp_star, verbose=True)
        assert count == 4
