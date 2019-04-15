from ..filters.all_filters import all_filters
from ..utils.misc import invert, values_map_to_same_key, one_hot
from ..utils.graph_ops import get_unspec_cover
from .alldiffs import count_alldiffs
import numpy as np
from functools import reduce

# TODO: count how many isomorphisms each background node participates in.
# TODO: switch from recursive to iterative implementation for readability

def recursive_isomorphism_counter(
        tmplt, world, *, unspec_cover, verbose, initial_changed_nodes):
    # If the node cover is empty, the unspec nodes are disconnected. Thus, we
    # can skip straight to counting solutions to the alldiff constraint problem
    if len(unspec_cover) == 0:
        # Elimination filter is not needed here and would be a waste of time
        all_filters(tmplt, world, verbose=False,
                    initial_changed_nodes=initial_changed_nodes)
        node_to_cands = {node: tmplt.get_cands(node) for node in tmplt.nodes}
        return count_alldiffs(node_to_cands)

    all_filters(tmplt, world, elimination=True, verbose=False,
                initial_changed_nodes=initial_changed_nodes)

    # Since the node cover is not empty, we first choose some valid
    # assignment of the unspecified nodes one at a time until the remaining
    # unspecified nodes are disconnected.
    n_isomorphisms = 0
    node_idx = unspec_cover[0]
    cand_idxs = np.argwhere(tmplt.is_cand[node_idx]).flat

    for i, cand_idx in enumerate(cand_idxs):
        tmplt_copy = tmplt.copy()
        tmplt_copy.is_cand[node_idx] = one_hot(cand_idx, tmplt.n_cands)

        # recurse to make assignment for the next node in the unspecified cover
        n_isomorphisms += recursive_isomorphism_counter(
            tmplt_copy, world, unspec_cover=unspec_cover[1:], verbose=verbose,
            initial_changed_nodes=one_hot(node_idx, tmplt.n_nodes))

        # TODO: more useful progress summary
        if verbose:
            print("depth {}: {} of {}".format(len(unspec_cover), i, len(cand_idxs)), n_isomorphisms)

    return n_isomorphisms


def count_isomorphisms(tmplt, world, verbose=True, filter_first=False):
    """
    counts the number of ways to assign template nodes to world nodes such that
    edges between template nodes also appear between the corresponding world
    nodes. Does not factor in the number of ways to assign the edges. Only
    counts the number of assignments between nodes.

    if the set of unspecified template nodes is too large or too densely
    connected, this code may never finish.
    """

    if filter_first:
        all_filters(tmplt, world, elimination=True, verbose=verbose)
        initial_changed_nodes = np.zeros(tmplt.nodes.shape)
    else:
        initial_changed_nodes = np.ones(tmplt.nodes.shape)

    unspec_cover = get_unspec_cover(tmplt)

    # Send zeros to initial_changed_nodes since we already just ran the filters
    return recursive_isomorphism_counter(
        tmplt, world, verbose=verbose, unspec_cover=unspec_cover,
        initial_changed_nodes=initial_changed_nodes)
