import numpy as np
from functools import reduce
from operator import mul

# TODO: parallelize?
# TODO: get set of values taken by tmplt edges?

# TODO: better name for this helper function
def iter_adj_pairs(tmplt, world):
    for channel, tmplt_adj in tmplt.ch_to_adj.items():
        world_adj = world.ch_to_adj[channel]
        yield (tmplt_adj, world_adj)
        yield (tmplt_adj.T, world_adj.T)

def topology_filter(tmplt, world, changed_nodes=None, **kwargs):
    """
    For each pair of neighbors in the template, ensure that any candidate for
    one neighbor has a corresponding candidate for the other neighbor to which
    it is connected by sufficiently many edges in each channel and direction.

    changed_nodes: boolean array indicating which nodes in the template have
                   candidates that have changed since last time this ran
    """
    for src_idx, dst_idx in tmplt.nbr_idx_pairs:
        if changed_nodes is not None:
            # If neither the source nor destination has changed, there is no
            # point in filtering on this pair of nodes
            if not (changed_nodes[src_idx] or changed_nodes[dst_idx]):
                continue

        # get indicators of candidate nodes in the world adjacency matrices
        src_is_cand = tmplt.is_cand[src_idx]
        dst_is_cand = tmplt.is_cand[dst_idx]

        # figure out which candidates have enough edges between them in world
        enough_edges = None
        for tmplt_adj, world_adj in iter_adj_pairs(tmplt, world):
            tmplt_adj_val = tmplt_adj[src_idx, dst_idx]

            # if there are no edges in this channel of the template, skip it
            if tmplt_adj_val == 0:
                continue

            # sub adjacency matrix corresponding to edges from the source
            # candidates to the destination candidates
            world_sub_adj = world_adj[:, dst_is_cand][src_is_cand, :]

            partial_enough_edges = world_sub_adj >= tmplt_adj_val
            if enough_edges is None:
                enough_edges = partial_enough_edges
            else:
                enough_edges = enough_edges.multiply(partial_enough_edges)

        # # i,j element is 1 if cands i and j have enough edges between them
        # enough_edges = reduce(mul, enough_edges_list, 1)

        # srcs with at least one reasonable dst
        src_matches = enough_edges.getnnz(axis=1) > 0
        tmplt.is_cand[src_idx][src_is_cand] = src_matches

        if src_idx != dst_idx:
            # dsts with at least one reasonable src
            dst_matches = enough_edges.getnnz(axis=0) > 0
            tmplt.is_cand[dst_idx][dst_is_cand] = dst_matches
