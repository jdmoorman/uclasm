from .misc import one_hot
import numpy as np

def get_node_cover(graph):
    """
    get a reasonably small set of nodes which, if removed, would cause
    all of the remaining nodes to become disconnected
    """

    cover = []

    # Initially there are no nodes in the cover. We add them one by one below.
    uncovered = np.ones(graph.n_nodes, dtype=np.bool_)

    # Until the cover disconnects the graph, add a node to the cover
    while graph.is_nbr[uncovered, :][:, uncovered].count_nonzero():

        # Add the uncovered node with the most neighbors
        nbr_counts = np.sum(graph.is_nbr[uncovered, :][:, uncovered], axis=0)

        imax = np.argmax(nbr_counts)
        node = graph.nodes[uncovered][imax]

        cover.append(graph.node_idxs[node])
        uncovered[uncovered] = ~one_hot(imax, np.sum(uncovered))

    return np.array(cover)
