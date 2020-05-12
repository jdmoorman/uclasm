import numpy as np
import scipy.sparse as sparse

from uclasm import Graph

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
