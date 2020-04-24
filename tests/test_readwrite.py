"""Tests for readwrite."""
import os
import pytest
import uclasm


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
        test_graph = uclasm.load_igraph(os.path.join(datadir, "aids.igraph"))
