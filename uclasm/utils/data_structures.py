"""
Filtering algorithms expect data to come in the form of Template, World
objects.
"""

from .misc import index_map
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import numpy as np
import networkx as nx
import pandas as pd
import functools
import time

# TODO: get rid of _GraphWithCandidates and filter related data stuff
# TODO: bring back functions needed for neighborhood filter

class Graph:
    def __init__(self, nodes, channels, adjs, labels=None):
        self.nodes = np.array(nodes)
        self.n_nodes = len(nodes)
        self.node_idxs = index_map(self.nodes)
        self.ch_to_adj = {ch: adj for ch, adj in zip(channels, adjs)}

        if labels is None:
            labels = [None]*len(nodes)
        self.labels = np.array(labels)


        self._composite_adj = None
        self._sym_composite_adj = None
        self._is_nbr = None

    @property
    def composite_adj(self):
        if self._composite_adj is None:
            self._composite_adj = sum(self.ch_to_adj.values())

        return self._composite_adj

    @property
    def sym_composite_adj(self):
        if self._sym_composite_adj is None:
            self._sym_composite_adj = self.composite_adj + self.composite_adj.T

        return self._sym_composite_adj

    @property
    def is_nbr(self):
        if self._is_nbr is None:
            self._is_nbr = self.sym_composite_adj > 0

        return self._is_nbr

    @property
    def channels(self):
        return self.ch_to_adj.keys()

    @property
    def adjs(self):
        return self.ch_to_adj.values()

    @property
    def nbr_idx_pairs(self):
        """
        Returns a 2d array with 2 columns. Each row contains the node idxs of
        a pair of neighbors in the graph. Each pair is only returned once, so
        for example only one of (0,3) and (3,0) could appear as rows.
        """
        return np.argwhere(sparse.tril(self.is_nbr))

    def subgraph(self, node_idxs):
        """
        Returns the subgraph induced by candidates
        """

        # throw out nodes not belonging to the desired subgraph
        nodes = self.nodes[node_idxs]
        labels = self.labels[node_idxs]
        adjs = [adj[node_idxs, :][:, node_idxs] for adj in self.adjs]

        # Return a new graph object for the induced subgraph
        return self.__class__(nodes, self.channels, adjs, labels=labels)

    def copy(self):
        """
        The only thing this bothers to copy is the adjacency matrices
        """
        return self.__class__(
            self.nodes, self.channels, [adj.copy() for adj in self.adjs],
            labels=self.labels)
