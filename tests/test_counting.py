"""Tests for the counting."""
import pytest
import uclasm
from uclasm.counting import count_alldiffs

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
    def test_count_isomorphisms(self):
        pass
