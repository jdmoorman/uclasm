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
    ch_to_world_adj_T = {channel: world_adj.T[:,:] for channel, world_adj in world.ch_to_adj.items()}
    for src_idx, dst_idx in np.argwhere(tmplt.is_nbr):
        if changed_cands is not None:
            # If neither the source nor destination has changed, there is no
            # point in filtering on this pair of nodes
            if not (changed_cands[src_idx] or changed_cands[dst_idx]):
                continue

        # get indicators of candidate nodes in the world adjacency matrices
        src_is_cand = candidates[src_idx]
        dst_is_cand = candidates[dst_idx]

        # enough_edges = np.zeros((len(src_is_cand), len(dst_is_cand)), dtype=np.bool)
        enough_edges = None
        # figure out which candidates have enough edges between them in world
        for channel in tmplt.ch_to_adj:
            tmplt_adj = tmplt.ch_to_adj[channel]
            world_adj = world.ch_to_adj[channel]

            tmplt_adj_val1 = tmplt_adj[src_idx, dst_idx]
            tmplt_adj_val2 = tmplt_adj[dst_idx, src_idx]

            # if there are no edges in this channel of the template, skip it
            if tmplt_adj_val1 == 0 and tmplt_adj_val2 == 0:
                continue
            world_adj_T = ch_to_world_adj_T[channel]
            if tmplt_adj_val1 > 0:
                # sub adjacency matrix corresponding to edges from the source
                # candidates to the destination candidates
                world_sub_adj = world_adj[:, dst_is_cand][src_is_cand, :]

                if enough_edges is None:
                    enough_edges = world_sub_adj >= tmplt_adj_val1
                else:
                    # enough_edges[world_sub_adj < tmplt_adj_val1] = False
                    enough_edges = enough_edges.multiply(world_sub_adj >= tmplt_adj_val1)
            if tmplt_adj_val2 > 0:
                world_sub_adj = world_adj_T[:, dst_is_cand][src_is_cand, :]

                if enough_edges is None:
                    enough_edges = world_sub_adj >= tmplt_adj_val2
                else:
                    # enough_edges[world_sub_adj < tmplt_adj_val2] = False
                    enough_edges = enough_edges.multiply(world_sub_adj >= tmplt_adj_val2)

        # i,j element is 1 if cands i and j have enough edges between them

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
