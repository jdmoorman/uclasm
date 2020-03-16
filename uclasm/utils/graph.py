"""
"""

from .misc import index_map
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import dask.dataframe as dd
from .misc import one_hot

class Graph:
    """An attributed multigraph class with support for attributes.

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

    """
    def __init__(self, adjs, channels=None, nodelist=None, edgelist=None):
        self.n_nodes = adjs[0].shape[0]
        self.n_channels = len(adjs)

        if channels is None:
            # e.g. ["channel 0", "channel 1", ...]
            channels = ["channel {}".format(i) for i in range(self.n_channels)]

        self.channels = list(channels)
        self.adjs = list(adjs)
        self.ch_to_adj = {ch: adj for ch, adj in zip(channels, adjs)}

        # If a nodelist is not supplied, generate a basic one
        if nodelist is None:
            node_names = ["node {}".format(i) for i in range(self.n_nodes)]
            nodelist = pd.DataFrame(node_names, columns=['node'])

        self.nodelist = nodelist
        self.nodes = self.nodelist['node']
        self.node_to_idx = index_map(self.nodes)

        # TODO: Make sure nodelist is indexed in a reasonable way
        # TODO: Make sure edgelist is indexed in a reasonable way

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
        either direction in any channel. The entry will be True if the nodes are
        connected by an edge in some channel and False otherwise.
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

    def subgraph(self, node_idxs):
        """Get the subgraph induced by the specified node indices.

        Parameters
        ----------
        node_idxs : 1darray()
            The indices corresponding to the nodes in the desired subgraph.

        Returns
        -------
        Graph
            The induced subgraph.
        """

        # throw out nodes not belonging to the desired subgraph
        adjs = [adj[node_idxs, :][:, node_idxs] for adj in self.adjs]
        nodelist = self.nodelist.iloc[node_idxs].reset_index(drop=True)
        nodes = nodelist['node']

        # TODO: require particular column names

        _srcs = self.edgelist['src'].isin(nodes)
        _dests = self.edgelist['dest'].isin(nodes)
        edgelist = self.edgelist[_srcs & _dests].reset_index(drop=True)

        # Return a new graph object for the induced subgraph
        return Graph(adjs, self.channels, nodelist, edgelist)

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

