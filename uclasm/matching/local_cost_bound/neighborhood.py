import numpy as np
import networkx as nx
from scipy import sparse
from scipy import optimize
import time

def get_edge_seqs(graph, channels=None):
    """
    Return the edge sequences in each channel for each node.

    Returns
    -------
    dict
        Keys are the node indexes and values are their corresponding edge seqs.
    """
    if channels is None:
        channels = graph.channels

    edge_seqs = {}
    for node_idx in range(graph.n_nodes):
        seq_list = []
        for channel in channels:
            adj = graph.ch_to_adj[channel]
            seq_list.extend([adj[node_idx,:], adj.T[node_idx,:]])
        edge_seqs[node_idx] = sparse.vstack(seq_list, format="csr")

    return edge_seqs

def neighborhood(smp):
    """
    Bound local assignment costs by neighborhood disagreements.

    If u is a node in the template and v is a node in the world, in order for
    v to be a candidate for u there should be a subgraph isomorphism from
    the neighborhood of u to the neighborhood of v. We can check if any such
    subgraph isomorphism exists in which the neighbors of v are candidates for
    the appropriate neighbors of u by looking for a bipartite matching.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute edgewise cost bounds.
    """
    # TODO: check whether a world node is a candidate for any tmplt node
    # ---> This can be achieved by reduce_world?

    tmplt_edge_seqs = get_edge_seqs(smp.tmplt)
    world_edge_seqs = get_edge_seqs(smp.world, channels=smp.tmplt.channels)

    nbr_counts = smp.tmplt.is_nbr.sum(axis=1).A.flatten()

    # TODO: might want candidates to be sparse in other filters
    sparse_is_cand = sparse.csr_matrix(smp.candidates())
    for tnode_idx, wnode_idx in np.transpose(sparse_is_cand.nonzero()):
        # If the template node has only 1 neighbor, the topology filter is
        # equivalent to the neighborhood filter, so there is no point in
        # using the neighborhood filter since it is more expensive.
        if nbr_counts[tnode_idx] == 1:
            continue

        tmplt_seq = tmplt_edge_seqs[tnode_idx]
        world_seq = world_edge_seqs[wnode_idx]

        # TODO: rewrite this part of neighborhood cost bound computation
        # to only take into account the matching in the tmplt node nbhd
        lap_mat_rows = []
        for tnbr_idx in smp.tmplt.is_nbr[tnode_idx].nonzero()[1]:
            # The world nodes that are not candidates to tnbr_idx are np.Inf
            lap_mat_row = smp.global_costs[tnbr_idx]
            lap_mat_row[lap_mat_row != np.Inf] = 0
            # Check if all of the necessary edges are present
            for i, edge_count in enumerate(tmplt_seq[:,tnbr_idx].A.flat):
                if edge_count > 0:
                    lap_mat_row += np.maximum(edge_count-world_seq[i].A, 0).flatten()
            lap_mat_rows.append(lap_mat_row)

        lap_mat = np.stack(lap_mat_rows)
        row_idxs, col_idxs = optimize.linear_sum_assignment(lap_mat)
        smp.local_costs[tnode_idx, wnode_idx] = lap_mat[row_idxs, col_idxs].sum()
