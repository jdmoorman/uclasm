"""Tests for the counting."""
import pytest
import uclasm
from uclasm.counting import count_alldiffs, count_isomorphisms, find_isomorphisms
from uclasm.matching.filters.run_filters import run_filters
from uclasm import Graph, ExactMatchingProblem

import numpy as np
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
    smp = ExactMatchingProblem(tmplt, world)
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
    smp = ExactMatchingProblem(tmplt, world)
    return smp

@pytest.fixture
def smp_node_cover():
    """Create a subgraph matching problem requiring the use of node cover."""
    adj0 = csr_matrix([[0, 1, 0, 1, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 0, 1, 0]])
    adj1 = csr_matrix([[0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0]])
    nodelist = pd.DataFrame(['a', 'b', 'c', 'd', 'e'], columns=[Graph.node_col])
    edgelist = pd.DataFrame([['a', 'b', 'c1'],
                             ['a', 'd', 'c1'],
                             ['e', 'b', 'c1'],
                             ['e', 'd', 'c1'],
                             ['b', 'c', 'c1'],
                             ['d', 'c', 'c1'],
                             ['a', 'c', 'c2'],
                             ['e', 'c', 'c2']], columns=[Graph.source_col,
                                                   Graph.target_col,
                                                   Graph.channel_col])
    tmplt = Graph([adj0, adj1], ['c1', 'c2'], nodelist, edgelist)
    world = Graph([adj0, adj1], ['c1', 'c2'], nodelist, edgelist)
    smp = ExactMatchingProblem(tmplt, world)
    return smp

@pytest.fixture
def smp_overlapping_cands():
    """Create a subgraph matching problem requiring proper handling when
    assigning groups of overlapping candidates."""
    tmplt_adj = csr_matrix([[0, 1, 1],
                            [1, 0, 0],
                            [0, 0, 0]])
    tmplt = Graph([tmplt_adj], ['c1'])
    world_adj = csr_matrix([[0, 1, 1, 1, 1],
                       [1, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
    world = Graph([world_adj], ['c1'])
    smp = ExactMatchingProblem(tmplt, world)
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
        run_filters(smp)
        assert np.sum(smp.candidates) == 3
        count = count_isomorphisms(smp, verbose=True)
        assert count == 1

    def test_find_isomorphisms(self, smp):
        run_filters(smp)
        iso_list = find_isomorphisms(smp, verbose=True)
        assert len(iso_list) == 1
        iso = iso_list[0]
        assert iso['a'] == 'a'
        assert iso['b'] == 'b'
        assert iso['c'] == 'c'

    def test_count_isomorphisms_star(self, smp_star):
        run_filters(smp_star)
        assert np.sum(smp_star.candidates) == 9
        count = count_isomorphisms(smp_star, verbose=True)
        assert count == 4

    def test_count_isomorphisms_node_cover(self, smp_node_cover):
        run_filters(smp_node_cover)
        assert np.sum(smp_node_cover.candidates) == 9
        count = count_isomorphisms(smp_node_cover, verbose=True)
        assert count == 4

    def test_count_isomorphisms_overlapping_cands(self, smp_overlapping_cands):
        run_filters(smp_overlapping_cands)
        assert np.sum(smp_overlapping_cands.candidates) == 7
        count = count_isomorphisms(smp_overlapping_cands, verbose=True)
        assert count == 6
