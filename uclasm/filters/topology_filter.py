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

def topology_filter(tmplt, world, candidates, *,
                    changed_cands=None, **kwargs):
    """
    For each pair of neighbors in the template, ensure that any candidate for
    one neighbor has a corresponding candidate for the other neighbor to which
    it is connected by sufficiently many edges in each channel and direction.

    changed_cands: boolean array indicating which nodes in the template have
                   candidates that have changed since last time this ran
    """
    for src_idx, dst_idx in tmplt.nbr_idx_pairs:
        if changed_cands is not None:
            # If neither the source nor destination has changed, there is no
            # point in filtering on this pair of nodes
            if not (changed_cands[src_idx] or changed_cands[dst_idx]):
                continue

        # get indicators of candidate nodes in the world adjacency matrices
        src_is_cand = candidates[src_idx]
        dst_is_cand = candidates[dst_idx]

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
        candidates[src_idx][src_is_cand] = src_matches
        if not any(src_matches):
            candidates[:,:] = False
            break

        if src_idx != dst_idx:
            # dsts with at least one reasonable src
            dst_matches = enough_edges.getnnz(axis=0) > 0
            candidates[dst_idx][dst_is_cand] = dst_matches
            if not any(dst_matches):
                candidates[:,:] = False
                break

    return tmplt, world, candidates


def topology_filter_dense(tmplt, world, candidates, 
                    noise_level=0, changed_cands=None, **kwargs):
    """
    For each pair of neighbors in the template, ensure that any candidate for
    one neighbor has a corresponding candidate for the other neighbor to which
    it is connected by sufficiently many edges in each channel and direction.

    changed_cands: boolean array indicating which nodes in the template have
                   candidates that have changed since last time this ran
    """
    for src_idx, dst_idx in tmplt.nbr_idx_pairs:
        # if enough noise to ignore all edges, don't bother trying to filter
        #if tmplt.sym_composite_adj[src_idx,dst_idx] <= noise_level:
            #continue

        if changed_cands is not None:
            # If neither the source nor destination has changed, there is no
            # point in filtering on this pair of nodes
            if not (changed_cands[src_idx] or changed_cands[dst_idx]):
                continue

        # get indicators of candidate nodes in the world adjacency matrices
        src_is_cand = candidates[src_idx]
        dst_is_cand = candidates[dst_idx]

        # figure out which candidates have enough edges between them in world
        edge_mismatch_total = None
        for tmplt_adj, world_adj in iter_adj_pairs(tmplt, world):
            tmplt_adj_val = tmplt_adj[src_idx, dst_idx]

            # if there are no edges in this channel of the template, skip it
            if tmplt_adj_val == 0:
                continue

            # sub adjacency matrix corresponding to edges from the source
            # candidates to the destination candidates
            world_sub_adj = world_adj[:, dst_is_cand][src_is_cand, :]

            edge_mismatch_count = np.maximum(0, tmplt_adj_val - world_sub_adj)
            if edge_mismatch_total is None:
                edge_mismatch_total = edge_mismatch_count
            else:
                edge_mismatch_total += edge_mismatch_count

        enough_edges = edge_mismatch_total <= noise_level

        # # i,j element is 1 if cands i and j have enough edges between them

        # srcs with at least one reasonable dst
        src_matches = enough_edges.sum(axis=1) > 0
        candidates[src_idx][src_is_cand] = src_matches
        if not any(src_matches):
            candidates[:,:] = False
            break

        if src_idx != dst_idx:
            # dsts with at least one reasonable src
            dst_matches = enough_edges.sum(axis=0) > 0
            candidates[dst_idx][dst_is_cand] = dst_matches
            if not any(dst_matches):
                candidates[:,:] = False
                break

    return tmplt, world, candidates
