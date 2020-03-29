"""TODO: Docstring."""


def edgewise_cost_bound(smp):
    """TODO: Docstring.

    TODO: Cite paper from REU.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute nodewise cost bounds.
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