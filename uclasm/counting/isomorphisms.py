from ..filters import run_filters, cheap_filters, all_filters
from ..utils.misc import invert, values_map_to_same_key, one_hot
from ..utils.graph_ops import get_node_cover
from .alldiffs import count_alldiffs
import numpy as np
from functools import reduce

# TODO: count how many isomorphisms each background node participates in.
# TODO: switch from recursive to iterative implementation for readability

def recursive_isomorphism_counter(tmplt, world, candidates, *,
                                  unspec_cover, verbose, init_changed_cands):
    # If the node cover is empty, the unspec nodes are disconnected. Thus, we
    # can skip straight to counting solutions to the alldiff constraint problem
    if len(unspec_cover) == 0:
        # Elimination filter is not needed here and would be a waste of time
        run_filters(tmplt, world, candidates=candidates, filters=cheap_filters,
                    verbose=False, init_changed_cands=init_changed_cands)
        node_to_cands = {node: world.nodes[candidates[idx]]
                         for idx, node in enumerate(tmplt.nodes)}
        return count_alldiffs(node_to_cands)

    run_filters(tmplt, world, candidates=candidates, filters=all_filters,
                verbose=False, init_changed_cands=init_changed_cands)

    # Since the node cover is not empty, we first choose some valid
    # assignment of the unspecified nodes one at a time until the remaining
    # unspecified nodes are disconnected.
    n_isomorphisms = 0
    node_idx = unspec_cover[0]
    cand_idxs = np.argwhere(candidates[node_idx]).flat

    for i, cand_idx in enumerate(cand_idxs):
        candidates_copy = candidates.copy()
        candidates_copy[node_idx] = one_hot(cand_idx, world.n_nodes)

        # recurse to make assignment for the next node in the unspecified cover
        n_isomorphisms += recursive_isomorphism_counter(
            tmplt, world, candidates_copy, unspec_cover=unspec_cover[1:],
            verbose=verbose, init_changed_cands=one_hot(node_idx, tmplt.n_nodes))

        # TODO: more useful progress summary
        if verbose:
            print("depth {}: {} of {}".format(len(unspec_cover), i, len(cand_idxs)), n_isomorphisms)

    return n_isomorphisms


def count_isomorphisms(tmplt, world, *, candidates=None, verbose=True):
    """
    counts the number of ways to assign template nodes to world nodes such that
    edges between template nodes also appear between the corresponding world
    nodes. Does not factor in the number of ways to assign the edges. Only
    counts the number of assignments between nodes.

    if the set of unspecified template nodes is too large or too densely
    connected, this code may never finish.
    """

    if candidates is None:
        tmplt, world, candidates = uclasm.run_filters(
            tmplt, world, filters=uclasm.all_filters, verbose=True)

    unspec_nodes = np.where(candidates.sum(axis=1) > 1)[0]
    unspec_cover = get_node_cover(tmplt.subgraph(unspec_nodes))

    # Send zeros to init_changed_cands since we already just ran the filters
    return recursive_isomorphism_counter(
        tmplt, world, candidates, verbose=verbose, unspec_cover=unspec_cover,
        init_changed_cands=np.zeros(tmplt.nodes.shape, dtype=np.bool))
