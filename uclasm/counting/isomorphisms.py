from ..filters import run_filters, cheap_filters, all_filters
from ..utils.misc import invert, values_map_to_same_key, one_hot
from ..utils.graph_ops import get_node_cover
from .alldiffs import count_alldiffs
import numpy as np
from functools import reduce

# TODO: count how many isomorphisms each background node participates in.
# TODO: switch from recursive to iterative implementation for readability
n_iterations = 0

def recursive_isomorphism_counter(tmplt, world, candidates, *,
                                  unspec_cover, verbose, init_changed_cands, count_iterations=False):
    global n_iterations
    n_iterations += 1
    # If the node cover is empty, the unspec nodes are disconnected. Thus, we
    # can skip straight to counting solutions to the alldiff constraint problem
    if len(unspec_cover) == 0:
        # Elimination filter is not needed here and would be a waste of time
        tmplt, world, candidates = run_filters(tmplt, world, candidates=candidates, filters=cheap_filters,
                    verbose=False, init_changed_cands=init_changed_cands)
        node_to_cands = {node: world.nodes[candidates[idx]]
                         for idx, node in enumerate(tmplt.nodes)}
        return count_alldiffs(node_to_cands)

    tmplt, world, candidates = run_filters(tmplt, world, candidates=candidates, filters=all_filters,
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
            verbose=verbose, init_changed_cands=one_hot(node_idx, tmplt.n_nodes), count_iterations=count_iterations)

        # TODO: more useful progress summary
        if verbose:
            print("depth {}: {} of {}".format(len(unspec_cover), i, len(cand_idxs)), n_isomorphisms)

    return n_isomorphisms


def count_isomorphisms(tmplt, world, *, candidates=None, verbose=True, count_iterations=False):
    """
    counts the number of ways to assign template nodes to world nodes such that
    edges between template nodes also appear between the corresponding world
    nodes. Does not factor in the number of ways to assign the edges. Only
    counts the number of assignments between nodes.

    if the set of unspecified template nodes is too large or too densely
    connected, this code may never finish.
    """
    global n_iterations
    n_iterations = 0

    if candidates is None:
        tmplt, world, candidates = uclasm.run_filters(
            tmplt, world, filters=uclasm.all_filters, verbose=verbose)

    unspec_nodes = np.where(candidates.sum(axis=1) > 1)[0]
    tmplt_subgraph = tmplt.subgraph(unspec_nodes)
    unspec_cover = get_node_cover(tmplt_subgraph)
    unspec_cover_nodes = [tmplt_subgraph.nodes[node_idx] for node_idx in unspec_cover]
    unspec_cover_idxes = [tmplt.node_idxs[node] for node in unspec_cover_nodes]

    # Send zeros to init_changed_cands since we already just ran the filters
    count = recursive_isomorphism_counter(
        tmplt, world, candidates, verbose=verbose, unspec_cover=unspec_cover_idxes,
        init_changed_cands=np.zeros(tmplt.nodes.shape, dtype=np.bool), count_iterations=count_iterations)
    if count_iterations:
        return count, n_iterations
    else:
        return count

def recursive_isomorphism_finder(tmplt, world, candidates, *,
                                 unspec_node_idxs, verbose, init_changed_cands,
                                 found_isomorphisms):
    if len(unspec_node_idxs) == 0:
        # All nodes have been assigned, add the isomorphism to the list
        new_isomorphism = {}
        for tmplt_idx, tmplt_node in enumerate(tmplt.nodes):
            if verbose:
                print(str(tmplt_node)+":", world.nodes[candidates[tmplt_idx]])
            new_isomorphism[tmplt_node] = world.nodes[candidates[tmplt_idx]][0]
        found_isomorphisms.append(new_isomorphism)
        return found_isomorphisms

    tmplt, world, candidates = run_filters(tmplt, world, candidates=candidates,
                filters=all_filters, verbose=False,
                init_changed_cands=init_changed_cands)

    node_idx = unspec_node_idxs[0]
    cand_idxs = np.argwhere(candidates[node_idx]).flat

    for i, cand_idx in enumerate(cand_idxs):
        candidates_copy = candidates.copy()
        candidates_copy[node_idx] = one_hot(cand_idx, world.n_nodes)

        # recurse to make assignment for the next node in the unspecified cover
        recursive_isomorphism_finder(
            tmplt, world, candidates_copy,
            unspec_node_idxs=unspec_node_idxs[1:],
            verbose=verbose,
            init_changed_cands=one_hot(node_idx, tmplt.n_nodes),
            found_isomorphisms=found_isomorphisms)
    return found_isomorphisms

def find_isomorphisms(tmplt, world, *, candidates=None, verbose=True):
    """ Returns a list of isomorphisms as dictionaries mapping template nodes to
    world nodes. Note: this is much slower than counting, and should only be
    done for small numbers of isomorphisms and fully filtered candidate matrices
    """
    if candidates is None:
        tmplt, world, candidates = uclasm.run_filters(
            tmplt, world, filters=uclasm.all_filters, verbose=verbose)
    unspec_node_idxs = np.where(candidates.sum(axis=1) > 1)[0]
    found_isomorphisms = []

    return recursive_isomorphism_finder(
        tmplt, world, candidates, verbose=verbose,
        unspec_node_idxs=unspec_node_idxs,
        init_changed_cands=np.zeros(tmplt.nodes.shape, dtype=np.bool),
        found_isomorphisms=found_isomorphisms)

def print_isomorphisms(tmplt, world, *, candidates=None, verbose=True):
    """ Prints the list of isomorphisms """
    print(find_isomorphisms(tmplt, world, candidates=candidates,
                            verbose=verbose))
