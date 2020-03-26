"""Graph class for representing networks with node and edge attributes."""

import scipy.sparse as sparse
import numpy as np
import pandas as pd

from .utils import index_map, one_hot


class Graph:
    """An multigraph class with support for node and edge attributes.

    Parameters
    ----------
    adjs : list(spmatrix)
        Adjacency matrices counting edges in each channel.
    channels : list(str), optional
        Types of edge, one per adjacency matrix.
    nodelist : DataFrame, optional
        Attributes of each node.
    edgelist : DataFrame, optional
        Attributes of each edge.

    Attributes
    ----------
    n_nodes : int
        Number of nodes in the graph.
    n_channels : int
        Number of types of edge in the graph.
    channels : list(str)
        Types of edge present in the graph.
    adjs : list(spmatrix)
        Adjacency matrices corresponding to each channel.
    ch_to_adj : dict(str, spmatrix)
        A map from channel names to corresponding adjacency matrices.
    nodes : Series
        A Series containing node identifiers. These are particularly
        useful for keeping track of nodes when taking subgraphs.
    nodelist : DataFrame
        A DataFrame containing node attribute information.
    edgelist : DataFrame, optional
        A DataFrame containing edge attribute information.
    node_col : str
        Column name for node identifiers in the nodelist.
    source_col : str
        Column name for source node identifiers in the edgelist.
    target_col : str
        Column name for target node identifiers in the edgelist.
    channel_col : str
        Column name for edge type/channel identifiers in the edgelist.
    """

    node_col = "Node"
    source_col = "Source"
    target_col = "Target"
    channel_col = "eType"

    def __init__(self, adjs, channels=None, nodelist=None, edgelist=None):
        """TODO: Docstring."""
        self.n_nodes = adjs[0].shape[0]
        self.n_channels = len(adjs)

        if channels is None:
            # e.g. ["channel 0", "channel 1", ...]
            channels = ["channel {}".format(i) for i in range(self.n_channels)]

        self.channels = list(channels)
        self.adjs = list(adjs)
        self.ch_to_adj = dict(zip(channels, adjs))

        # TODO: If an edgelist was supplied, use it to compute this stuff.

        # If a nodelist is not supplied, generate a basic one
        if nodelist is None:
            # e.g. ["node 0", "node 1", ...]
            node_names = ["node {}".format(i) for i in range(self.n_nodes)]
            nodelist = pd.DataFrame(node_names, columns=[self.node_col])

        self.nodelist = nodelist
        self.nodes = self.nodelist[self.node_col]
        self.node_to_idx = index_map(self.nodes)

        # TODO: Make sure nodelist is indexed in a reasonable way
        # TODO: Make sure edgelist is indexed in a reasonable way
        # TODO: Check dtypes of edgelist and nodelist columns.

        self.edgelist = edgelist

    @property
    def composite_adj(self):
        """spmatrix: Composite adjacency matrix of the graph.

        Each entry of this matrix corresponds to the total number of edges
        of any type going from the node corresponding to the row to the node
        corresponding to the column.
        """
        if not hasattr(self, "_composite_adj"):
            self._composite_adj = sum(self.ch_to_adj.values())

        return self._composite_adj

    @property
    def sym_composite_adj(self):
        """spmatrix: Symmetrized composite adjacency matrix of the graph.

        Each entry of this matrix corresponds to the total number of edges
        of any type between the pair of nodes indicated by the row and column
        indices, ignoring the direction of the edges.
        """
        if not hasattr(self, "_sym_composite_adj"):
            self._sym_composite_adj = self.composite_adj + self.composite_adj.T

        return self._sym_composite_adj

    @property
    def is_nbr(self):
        """spmatrix: Boolean adjacency matrix of the graph.

        Each entry of this matrix indicates whether the pair of nodes
        corresponding to the row and column indices are connected by an edge in
        either direction in any channel. The entry will be True if the nodes
        are connected by an edge in some channel and False otherwise.
        """
        if not hasattr(self, "_is_nbr"):
            self._is_nbr = self.sym_composite_adj > 0

        return self._is_nbr

    @property
    def nbr_idx_pairs(self):
        """2darray: A [N, 2] array of adjacent pairs of node indices.

        A 2d array with 2 columns. Each row contains the indices of a pair of
        neighboring nodes in the graph. Each pair is only returned once, so
        only one of (i, j) and (j, i) can appear as rows.
        """
        return np.argwhere(sparse.tril(self.is_nbr))

    @property
    def features(self):
        """2darray: An [n_features, n_nodes] array of node features.

        TODO: Add clustering coefficient: The number of edges between adjacent
        nodes (not including edges to the node being evaluated).

        A 2darray of shape [3*n_channels, n_nodes] providing the in-degree,
        out-degree, and number of self edges for each node in the graph.
        """
        if not hasattr(self, "_features"):
            features_list = []

            # For each channel, compute some features.
            for adj in self.adjs:
                self_edges = np.reshape(adj.diagonal(), (1, -1))
                in_degrees = adj.sum(axis=0).A - self_edges
                out_degrees = adj.sum(axis=1).T.A - self_edges

                features_list.extend([self_edges, in_degrees, out_degrees])

            self._features = np.concatenate(features_list, axis=0)

        return self._features

    def rename_nodes(self, old_to_new):
        """Rename nodes inplace.

        TODO: Test this function.

        Parameters
        ----------
        old_to_new : dict(str, str)
            Map from old node names to new node names
        """
        self.nodelist[[self.node_col]] = self.nodelist[[self.node_col]]\
            .applymap(self.node_to_idx.get)

        node_cols = [self.source_col, self.target_col]
        self.edgecounts[node_cols] = self.edgecounts[node_cols]\
            .applymap(self.node_to_idx.get)

    def node_subgraph(self, node_idxs):
        """Get the subgraph induced by the specified node indices.

        Parameters
        ----------
        node_idxs : 1darray
            The indices corresponding to the nodes in the desired subgraph.

        Returns
        -------
        Graph
            The induced subgraph.
        """
        # throw out nodes not belonging to the desired subgraph
        adjs = [adj[node_idxs, :][:, node_idxs] for adj in self.adjs]
        nodelist = self.nodelist.iloc[node_idxs].reset_index(drop=True)
        nodes = nodelist[self.node_col]

        # TODO: Test this to see if it works with dask DataFrames.
        _sources = self.edgelist[self.source_col].isin(nodes)
        _targets = self.edgelist[self.target_col].isin(nodes)
        edgelist = self.edgelist[_sources & _targets].reset_index(drop=True)

        # Return a new graph object for the induced subgraph
        return Graph(adjs, self.channels, nodelist, edgelist)

    def channel_subgraph(self, channels):
        """Get the subgraph induced by the specified channels.

        Parameters
        ----------
        channels : 1darray
            The desired channels to keep in the subgraph.

        Returns
        -------
        Graph
            The induced subgraph.
        """
        # throw out nodes not belonging to the desired subgraph
        adjs = [self.ch_to_adj[ch] for ch in channels]

        # Drop edges that do not have types among the desired channels.
        edge_ind = self.edgelist[self.channel_col].isin(channels)
        edgelist = self.edgelist[edge_ind].reset_index(drop=True)

        # Return a new graph object for the induced subgraph
        return Graph(adjs, channels, self.nodelist, edgelist)

    def node_cover(self):
        """Get the indices of nodes for a node cover, sorted by importance.

        This function provides no warranty of the optimality of the node cover.
        The computed node cover may be far from the smallest possible.

        Returns
        -------
        1darray
            The indices of a set of nodes in a node cover.
        """
        cover = []

        # Initially there are no nodes in the cover. Thus all of the nodes in
        # the graph are uncovered.
        uncov = np.ones(self.n_nodes, dtype=np.bool_)

        # Until the cover disconnects the graph, add a node to the cover
        while self.is_nbr[uncov, :][:, uncov].nnz:

            # Add the uncov node with the most neighbors
            nbr_counts = np.sum(self.is_nbr[uncov, :][:, uncov], axis=0)

            imax = np.argmax(nbr_counts)
            cover.append(imax)

            # Cover the node corresponding to imax.
            uncov[uncov] = ~one_hot(imax, np.sum(uncov))

        # TODO: Remove any nodes from the node cover which are not necessary.

        return np.array(cover)
