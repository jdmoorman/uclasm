"""
Filtering algorithms expect data to come in the form of Graph objects
"""

from ..equivalence.partition_sparse import bfs_partition_graph
from .misc import index_map
import scipy.sparse as sparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import graphviz as gv

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
        self._eq_classes = None

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

    @property
    def out_degree(self):
        if self.out_degree_array is not None:
            return self.out_degree_array
        else:
            self.out_degree_array = {ch: self.ch_to_adj[ch].sum(axis=1)
                                     for ch in self.channels}
            return self.out_degree_array

    @property
    def in_degree(self):
        if self.in_degree_array is not None:
            return self.in_degree_array
        else:
            self.in_degree_array = {ch: self.ch_to_adj[ch].sum(axis=0)
                                     for ch in self.channels}
            return self.in_degree_array

    @property
    def degree(self):
        if self.degree_array is not None:
            return self.degree_array
        else:
            self.degree_array = {channel: self.in_degree[channel] 
                                          + self.out_degree[channel]
                                 for channel in self.channels}
            return self.degree_array

    @property
    def neighbors(self):
        if self.neighbors_list:
            return self.neighbors_list
        else:
            self.compute_neighbors()
        return self.neighbors_list

    def compute_neighbors(self):
        for i in range(self.n_nodes):
            # We grab first element since nonzero returns a tuple of 1 element
            if sparse.issparse(self.sym_composite_adj):
                self.neighbors_list.append(self.sym_composite_adj[i].nonzero()[1])
            else:
                self.neighbors_list.append(self.sym_composite_adj[i].nonzero()[0])

    @property
    def eq_classes(self):
        """
        Return the Equivalence object associated with the graph.
        """
        if self._eq_classes is None:
            self._eq_classes = bfs_partition_graph(self.ch_to_adj)
        return self._eq_classes

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

    def sparsify(self):
        """
        Converts the stored adjacency matrices into sparse matrices in the
        csr_matrix format
        """
        if self.is_sparse:
            return
        self._is_nbr = sparse.csr_matrix(self.is_nbr)
        self._sym_composite_adj = sparse.csr_matrix(self.sym_composite_adj)
        self._composite_adj = sparse.csr_matrix(self.composite_adj)
        for ch, adj in self.ch_to_adj.items():
            self.ch_to_adj[ch] = sparse.csr_matrix(adj)
        self.is_sparse = True

    def densify(self):
        """
        Converts the stored adjacency matrices into standard arrays.
        This only affects the matrices in self.ch_to_adj, not any other
        possible sparse representations of data.

        This will cause an error if the matrices are already dense.
        """
        if not self.is_sparse:
            return
        self._is_nbr = self.is_nbr.A
        self._sym_composite_adj = self.sym_composite_adj.A
        self._composite_adj = self.composite_adj.A
        for ch, adj in self.ch_to_adj.items():
            self.ch_to_adj[ch] = adj.A
        self.is_sparse = False

    def copy(self):
        """
        The only thing this bothers to copy is the adjacency matrices
        """
        return Graph(self.nodes, self.channels,
                     [adj.copy() for adj in self.adjs],
                     labels=self.labels)

    def get_node_cover(self):
        """
        get a reasonably small set of nodes which, if removed, would cause
        all of the remaining nodes to become disconnected
        """

        cover = []

        # Initially there are no nodes in the cover. 
        # We add them one by one below.
        uncovered = np.ones(self.n_nodes, dtype=np.bool_)

        # Until the cover disconnects the self, add a node to the cover
        while self.is_nbr[uncovered, :][:, uncovered].count_nonzero():

            # Add the uncovered node with the most neighbors
            nbr_counts = np.sum(self.is_nbr[uncovered, :][:, uncovered], axis=0)

            imax = np.argmax(nbr_counts)
            node = self.nodes[uncovered][imax]

            cover.append(self.node_idxs[node])
            uncovered[uncovered] = ~one_hot(imax, np.sum(uncovered))

        return np.array(cover)

    def to_simple_graph(self):
        """
        This will construct a simple directed graph by replacing any multichannel 
        multiedges with a single node which connects to both incident nodes
        with a label indicating number of edges in each channel.

        Returns:
            Graph: The constructed simple graph
        """
        
    def write_to_file(self, filename):
        """
        Writes the graph out in the following format:

        <Graph Name>
        <# Nodes>
        <# Channels>
        <Channel1>
        <# Edges in Channel1>
        <From1> <To1> <Count1>
        ...
        <FromN> <ToN> <CountN>
        <Channel2>
        <# Edges in Channel1>
        ...

        where <Fromi>, <Toi> <Counti> are the index of the source node,
        the index of the destination node, and the count of edges for the
        i-th edge in a given channel.
        """
        with open(filename, 'w') as f:
            f.write('{}\n'.format(self.name))
            f.write('{}\n'.format(self.n_nodes))
            f.write('{}\n'.format(len(list(self.channels))))
            for channel in self.channels:
                f.write('{}\n'.format(channel))
                f.write('{}\n'.format(self.get_n_edges()[channel]))
                for _, fro, to, count in self.edge_iterator(channel):
                    f.write('{} {} {}\n'.format(fro, to, count))

    def channel_to_networkx_graph(self, channel):
        """
        Convert the given channel into a networkx MultiDiGraph.
        """
        return nx.from_scipy_sparse_matrix(self.ch_to_adj[channel],
                                           parallel_edges=True)

    def to_networkx_graph(self):
        """
        Return a dictionary mapping channels to networkx MultiDiGraphs.
        """
        return {channel: self.channel_to_networkx_graph(channel)
                         for channel in self.channels}

    def to_networkx_composite_graph(self):
        """
        Return a networkx-style MultiDiGraph from the sum of the adjacency
        matrices
        """
        comp_matrix = sum(self.ch_to_adj.values())
        return nx.from_scipy_sparse_matrix(comp_matrix, parallel_edges=True,
                                           create_using=nx.MultiDiGraph)

    def gv_graph(self):
        """
        Construct a graph using graphviz and returns it. This is is a
        visualization object and therefore has attributes associated to how
        the graph looks. Each channel will be colored differently.

        Returns:
            gv.Digraph: The constructed graphviz Digraph
        """
        cmap = plt.get_cmap('Set1')

        gv_graph = gv.Digraph()
        for i in range(self.n_nodes):
            node_name = self.nodes[i]
            if self.labels[i]:
                gv_graph.node(str(node_name), label=self.labels[i])
            else:
                gv_graph.node(str(node_name))

        for i, channel in enumerate(self.channels):
            color = matplotlib.colors.rgb2hex(cmap.colors[i])

            dok_matrix = self.ch_to_adj[channel].todok()
            for ((v1_index, v2_index), count) in dok_matrix.items():
                v1_name = str(self.nodes[v1_index])
                v2_name = str(self.nodes[v2_index])
                gv_graph.edge(v1_name, v2_name, color=color)

        return gv_graph

    def plot_composite_graph(self, axis=None, **kwargs):
        """
        Plot the graph onto the given axis. This will plot the composite graph.
        """
        comp_graph = self.to_networkx_composite_graph()
        layout = nx.kamada_kawai_layout(comp_graph)
        
        if axis is None:
            fig, axis = plt.subplots()
        
        nx.draw(comp_graph, pos=layout, ax=axis, **kwargs)
        
        return axis

    def plot_equivalence_classes(self, equivalence, axis=None, **kwargs):
        """
        Plot the graph with the nodes of the same equivalence class colored
        the same. This will plot the composite graph for ease of visualization.
        This can handle at most 8 equivalence classes.

        Args:
            equivalence (Equivalence): An equivalence structure on the graph.
                The elements should be numbers corresponding to indices of
                the graph vertices
            axis (plt.Axis): Optionally, an axis to plot to
        Returns:
            The axis on which the graph is plotted
        """

        # This map has 8 different colors.
        cmap = plt.get_cmap('Set1')
        # We use the color gray for any trivial equivalence class.
        GRAY_INDEX = 8

        color_map = [""] * self.n_nodes
        count = 0
        classes = equivalence.classes()
        for eq_class in classes.values():
            if len(eq_class) == 1:
                for elem in eq_class:
                    color_map[elem] = cmap.colors[GRAY_INDEX]
            else:
                for elem in eq_class:
                    color_map[elem] = cmap.colors[count]
                count += 1
        
        comp_graph = self.to_networkx_composite_graph()
        layout = nx.kamada_kawai_layout(comp_graph)
        
        if axis is None:
            fig, axis = plt.subplots()
        
        nx.draw(comp_graph, pos=layout, ax=axis, node_color=color_map,
                **kwargs)
        
        return axis

    def threshold(self, graph):
        """
        This function should threshold the max number of edges in this graph
        by the maximal number of edges in the passed in graph in each channel
        """
        for channel in self.channels:
            adj_mat = self.ch_to_adj[channel]
            graph_max = graph.ch_to_adj[channel].max()
            adj_mat.data = np.minimum(adj_mat.data, graph_max)

    def to_simple_graph(self):
        """
        Convert the graph into a simple directed graph. It does this by
        taking any pair of adjacent nodes and replacing the edges between
        them by a node adjacent to both with a label specifying how many
        edges in each channel.
        """
        # Convert each adj to dok format for fast indexing
        dok_adjs = {ch: adj.todok() for (ch, adj) in self.ch_to_adj.items()}

        nbr_idx_pairs = np.argwhere(self.composite_adj > 0)
        n_pairs = nbr_idx_pairs.shape[0]
        new_n_nodes = self.n_nodes + n_pairs

        new_adj = sparse.dok_matrix((new_n_nodes, new_n_nodes), dtype=np.bool)

        labels = list(self.labels)
        for i in range(n_pairs):
            start_idx, end_idx = nbr_idx_pairs[i,:]

            # The index of the node this pair corresponds to
            pair_idx = self.n_nodes + i

            # We set the adjacency relationship here
            new_adj[start_idx, pair_idx] = 1
            new_adj[pair_idx, end_idx] = 1

            # Here we construct the label for the pair.
            label_cpts = []
            for ch, adj in dok_adjs.items():
                edge_count = adj[start_idx, end_idx]
                if edge_count > 0:
                    label_cpts.append("{}:{}".format(ch, edge_count))
            labels.append(','.join(label_cpts))
        
        new_adj = new_adj.tocsr()
        graph = Graph(list(range(new_n_nodes)), [0], [new_adj], labels=labels) 
        return graph, nbr_idx_pairs

def read_from_file(filename):
    """
    Reads in a multichannel graph from a file in the format specified in
    write_to_file. This is the file specified in the Solnon benchmarks
    """
    with open(filename) as f:

        def getline():
            while True:
                line = f.readline()
                (line, *comment) = line.split('#')
                line = line.rstrip()
                if line:
                    return line

        name =  getline().rstrip()
        n_nodes = int(getline())
        n_channels = int(getline())
        channels = []
        adjs = []
        for i in range(n_channels):
            adj_mat = np.zeros((n_nodes, n_nodes))
            channel_name = getline().rstrip()
            channel_size = int(getline())
            seen = 0
            while seen < channel_size:
                line = getline()

                fro, to, count = list(map(int, line.split()))
                adj_mat[fro,to] = count
                seen += count
            channels.append(channel_name)
            adjs.append(sparse.csr_matrix(adj_mat))

        nodes = list(range(n_nodes))
        return Graph(nodes, channels, adjs, name=name)


def read_igraph_file(filename):
    """
    This function will read all graphs in an igraph file.

    Args:
        filename (str): The name of the file stored in igraph format
    Returns:
        list[Graph]: A list of Graphs stored in the file
    """
    graphs = []
    curr_vert_count = 0
    curr_vert_labels = []
    # A mapping from channel to adjacency matrix
    curr_adj_matrices = {}
    first = True
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            # This indicates we are starting a new graph
            if line.startswith('t'):
                if first:
                    first = False
                    continue
                else:
                    # Construct the Graph
                    verts = list(range(curr_vert_count))
                    channels = list(curr_adj_matrices.keys())
                    # We convert to csr format as that is standard Graph fmt.
                    adj_matrices = [curr_adj_matrices[ch].tocsr()
                                    for ch in channels]
                    graph = Graph(verts, channels, adj_matrices, 
                                  curr_vert_labels)
                    graphs.append(graph)
                    # Reset all the current values for new graph
                    curr_vert_count = 0
                    curr_vert_labels = []
                    curr_adj_matrices = {}
            elif line.startswith('v'):
                # Vertex line
                # Format "v <index> <label>"
                index, label = map(int, line.split()[1:])
                curr_vert_count += 1
                curr_vert_labels.append(label)
            elif line.startswith('e'):
                # Edge line
                # Format "e <start> <end> <label>"
                # Edges are assumed undirected
                start, end, label = map(int, line.split()[1:])
                if label not in curr_adj_matrices:
                    adj = sparse.dok_matrix((curr_vert_count, curr_vert_count), 
                                            dtype=np.int32)
                    curr_adj_matrices[label] = adj
                curr_adj_matrices[label][start,end] = 1
                curr_adj_matrices[label][end,start] = 1
        else:
            # Construct the final graph
            # This is necessary because once the last line is read, we exit
            # the loop.
            verts = list(range(curr_vert_count))
            channels = list(curr_adj_matrices.keys())
            adj_matrices = [curr_adj_matrices[ch] for ch in channels]
            graph = Graph(verts, channels, adj_matrices, 
                          curr_vert_labels)
            graphs.append(graph)

    return graphs
