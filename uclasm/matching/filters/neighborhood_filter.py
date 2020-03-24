import numpy as np
import networkx as nx
from scipy import sparse
from scipy import optimize
import time

# TODO: This filter is very slow. Make it faster.
# TODO: come up with terminology for "star neighborhood"
# TODO: utilize `changed_cands`

def get_edge_seqs(graph, channels=None, nodes=None):
    if channels is None:
        channels = graph.ch_to_adj.keys()

    if nodes is None:
        nodes = np.ones(graph.nodes.shape, dtype=np.bool)

    edge_seqs = {}
    for node_idx in range(graph.n_nodes):
        seq_list = []
        for channel in channels:
            adj = graph.ch_to_adj[channel]
            seq_list.extend([adj[node_idx,:], adj.T[node_idx,:]])
        edge_seqs[node_idx] = sparse.vstack(seq_list, format="csr")

    return edge_seqs

def neighborhood_filter(tmplt, world, candidates, *,
                        changed_cands=None, **kwargs):
    """
    If u is a node in the template and v is a node in the world, in order for
    v to be a candidate for u there should be a subgraph isomorphism from
    the neighborhood of u to the neighborhood of v. We can check if any such
    subgraph isomorphism exists in which the neighbors of v are candidates for
    the appropriate neighbors of u by looking for a bipartite matching.
    """
    # TODO: reduce copy paste against stats filter for is_cand_any

    # Boolean array indicating if a given world node is a candidate for any
    # template node. If a world node is not a candidate for any template nodes,
    # we shouldn't bother calculating its features.
    is_cand_any = np.any(candidates, axis=0)

    # No candidates for any template node
    if np.sum(is_cand_any) == 0:
        return

    tmplt_edge_seqs = get_edge_seqs(tmplt)
    world_edge_seqs = get_edge_seqs(world, channels=tmplt.channels,
                                    nodes=is_cand_any)

    nbr_counts = tmplt.is_nbr.sum(axis=1).A.flatten()

    # TODO: might want candidates to be sparse in other filters
    sparse_is_cand = sparse.csr_matrix(candidates)
    for tnode_idx, wnode_idx in np.transpose(sparse_is_cand.nonzero()):
        # If the template node has only 1 neighbor, the topology filter is
        # equivalent to the neighborhood filter, so there is no point in
        # using the neighborhood filter since it is more expensive.
        if nbr_counts[tnode_idx] == 1:
            continue

        tmplt_seq = tmplt_edge_seqs[tnode_idx]
        world_seq = world_edge_seqs[wnode_idx]

        lap_mat_rows = []
        for tnbr_idx in range(tmplt.n_nodes):
            lap_mat_row = sparse_is_cand[tnbr_idx]
            # Check if all off the necessary edges are present
            for i, edge_count in enumerate(tmplt_seq[:,tnbr_idx].A.flat):
                if edge_count > 0:
                    lap_mat_row = lap_mat_row.multiply(world_seq[i] >= edge_count)
            lap_mat_rows.append(lap_mat_row)

        # TODO: do we really need to run LAP with this whole matrix?
        # we should throw out empty rows and columns before densifying.
        lap_mat = ~sparse.vstack(lap_mat_rows).A
        row_idxs, col_idxs = optimize.linear_sum_assignment(lap_mat)
        if lap_mat[row_idxs, col_idxs].sum() > 0:
            candidates[tnode_idx, wnode_idx] = 0

    return tmplt, world, candidates
