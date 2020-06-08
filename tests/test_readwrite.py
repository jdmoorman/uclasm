"""Tests for readwrite."""
import os
import pytest
import uclasm
import numpy as np

@pytest.fixture
def datadir():
    return "tests/test_readwrite"

class TestLoadEdgelist:
    """Tests related to loading edgelists """
    def test_load_edgelist(self, datadir):
        tmplt = uclasm.load_edgelist(os.path.join(datadir, "template.csv"),
                                     file_source_col="Source",
                                     file_target_col="Target",
                                     file_channel_col="eType")
        world = uclasm.load_edgelist(os.path.join(datadir, "world.csv"),
                                     file_source_col="Source",
                                     file_target_col="Target",
                                     file_channel_col="eType")

class TestLoadIgraph:
    """Tests related to loading igraph files """
    def test_load_igraph(self, datadir):
        test_graphs = uclasm.load_igraph(os.path.join(datadir, "aids.igraph"))
        adj0_0 = [[0, 1, 1, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 0]]
        adj0_1 = [[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]]
        adj1_0 = [[0, 1, 0, 0],
                  [1, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 1, 0, 0]]
        adj1_1 = [[0, 0, 1, 0],
                  [0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 0]]

        assert test_graphs[0].adjs[0].A.tolist() == adj0_0
        assert test_graphs[0].adjs[1].A.tolist() == adj0_1
        assert test_graphs[1].adjs[0].A.tolist() == adj1_0
        assert test_graphs[1].adjs[1].A.tolist() == adj1_1
