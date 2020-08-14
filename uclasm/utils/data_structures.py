"""
Filtering algorithms expect data to come in the form of Graph objects
"""

from .misc import index_map
import scipy.sparse as sparse
import numpy as np
import networkx as nx

class Graph:
    def __init__(self, nodes, channels, adjs, labels=None):
        self.nodes = np.array(nodes)
        self.n_nodes = len(nodes)
        self.node_idxs = index_map(self.nodes)
        self.ch_to_adj = {ch: adj for ch, adj in zip(channels, adjs)}
        self.channels = channels
        self.adjs = adjs

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
        return Graph(nodes, self.channels, adjs, labels=labels)

    def copy(self):
        """
        The only thing this bothers to copy is the adjacency matrices
        """
        return Graph(self.nodes, self.channels,
                     [adj.copy() for adj in self.adjs],
                     labels=self.labels)

    def write_channel_solnon(self, filename, channel, dir=""):
        """
        Write out adjacency matrix for specific channel as adjacency list to
        a file in the Solnon format.
        """
        with open(filename, 'w') as f:
            f.write(str(self.n_nodes) + '\n')

            adj_mat = self.ch_to_adj[channel]
            curr_index = 0
            curr_adj = []
            for (node_index, adjacent_index) in zip(*adj_mat.nonzero()):
                if node_index != curr_index:
                    # We write out number of adjacent nodes, then the list
                    # of adjacent nodes.
                    f.write(" ".join(map(str, [len(curr_adj)]+curr_adj))+'\n')

                    for i in range(curr_index+1, node_index):
                        # For each index in between, these have no adjacent
                        # nodes, so we write zero for each of these nodes.
                        f.write("0\n")

                    curr_adj = []
                    curr_index = node_index

                curr_adj.append(adjacent_index)

            # Write out remaining nodes
            f.write(" ".join(map(str, [len(curr_adj)] + curr_adj)) + '\n')

            for i in range(curr_index+1, self.n_nodes):
                f.write('0\n')

    def write_file_solnon(self, filename):
        """
        Writes out the graph in solnon format. This format is described as
        follows:
        Each graph is described in a text file. If the graph has n vertices,
        then the file has n+1 lines:
            -The first line gives the number n of vertices.
            -The next n lines give, for each vertex, its number of successor
            nodes, followed by the list of its successor nodes.
        If there are multiple channels, then it will create one file for
        each channel.
        """

        if len(list(self.channels)) > 1:
            def add_channel_to_name(filename, channel):
                *name, ext = filename.split('.')
                name = '.'.join(name)
                new_name = name + '_' + str(channel) + '.' + ext
                return new_name

            filenames = [add_channel_to_name(filename, channel)
                         for channel in self.channels]

            for name, channel in zip(filenames, self.channels):
                self.write_channel_solnon(name, channel)
        else:
            # Extract sole channel
            channel = list(self.channels)[0]
            self.write_channel_solnon(filename, channel)

    def graph_to_txt(self, file_name, dir="."):
        """
        Transform an uclasm graph to "nodes.txt" and "edges.txt".

        Note: used for comparison with sumgra.

        TODO: Support multi-labels.
        """

        nodes_file = open("{}/{}_nodes.txt".format(file_name, dir), "w")
        for label in self.labels:
            if label:
                nodes_file.write(label+"\n")
            else:
                nodes_file.write("-1\n")
        nodes_file.close()
        print ("Total number of nodes is: {}".format(len(self.nodes)))

        # Note: channels need to be named 0, ..., (# channels-1)
        num_edges = 0
        edges_file = open("{}/{}_edges.txt".format(file_name, dir), "w")
        for src, dst in zip(self.is_nbr.nonzero()[0], self.is_nbr.nonzero()[1]):
            channels_with_edges = \
                [str(ch_idx) for ch_idx in range(len(self.channels)) \
                             if self.adjs[ch_idx][src, dst]]
            if len(channels_with_edges) > 0:
                num_edges += 1
                joined_string = ",".join(channels_with_edges)
                edges_file.write("{} {} ".format(src, dst)+joined_string+"\n")
        edges_file.close()
        print ("Total number of edges is: {}".format(num_edges))

    def graph_to_ged(self, file_name, dir="."):
        """
        Transform an uclasm graph to "nodes.txt" and "edges.txt".

        Note: used for comparison with Multi-RI.
        """
        file = open("{}/{}.txt".format(dir, file_name), "w")
        file.write(str(len(self.nodes))+"\n")
        for label in self.labels:
            if label:
                file.write(label+"\n")
            else:
                file.write("-1\n")
        print ("Total number of nodes is: {}".format(len(self.nodes)))

        # Note: channels need to be named 0, ..., (# channels-1)
        file.write(str(self.composite_adj.count_nonzero())+"\n")
        print ("Total number of edges is: {}".format(self.composite_adj.count_nonzero()))
        for src, dst in zip(self.is_nbr.nonzero()[0], self.is_nbr.nonzero()[1]):
            channels_with_edges = \
                [str(ch_idx) for ch_idx in range(len(self.channels)) \
                             if self.adjs[ch_idx][src, dst]]
            if len(channels_with_edges) > 0:
                joined_string = ",".join(channels_with_edges)
                file.write("{}\t{}\t".format(src, dst)+joined_string+"\n")
        file.close()

def from_networkx_graph(nx_graph):
    """
    This will convert a networkx style Graph into a uclasm style Graph.
    Currently this does not preserve node or edge labels. It just copies
    over the adjacency structure.

    Parameters
    ----------
    nx_graph : networkx.Graph
    """

    # TODO: Add the ability to port over node and edge labels
    adj = nx.to_scipy_sparse_matrix(nx_graph)
    nodes = list(range(nx_graph.number_of_nodes()))
    channels = [0]
    return Graph(nodes, channels, [adj])
