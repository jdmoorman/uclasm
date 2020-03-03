from ..filters import run_filters, cheap_filters, all_filters
from ..utils.misc import invert, values_map_to_same_key, one_hot
from ..utils.graph_ops import get_node_cover
from .alldiffs import count_alldiffs
import numpy as np
from functools import reduce

# TODO: count how many isomorphisms each background node participates in.
# TODO: switch from recursive to iterative implementation for readability

def pick_minimum_domain_vertex(candidates):
    """
    This function will return the index of the template vertex with the
    smallest domain that has at least 2 candidates. If there are multiple
    minima, pick the one with the smallest index.

    Args:
        candidates (np.array): The candidate matrix
    Returns:
        int: The index of the template vertex
    """
    # Count the number of candidates for each.
    cand_counts = candidates.sum(axis=1)
    # Matched vertices are those with 1 candidate.
    matched_t_verts = cand_counts == 1

    # We set all matched vertex counts to be high enough so that they
    # will not be chosen.
    _, n_world_verts = candidates.shape
    cand_counts[matched_t_verts] = n_world_verts + 1
    t_vert = np.argmin(cand_counts)

    return t_vert

def recursive_isomorphism_counter(tmplt, world, candidates, *,
        unspec_cover, verbose, init_changed_cands, tmplt_equivalence=False, 
        world_equivalence=False):
    """
    Recursive routine for solving subgraph isomorphism.

    Args:
        tmplt (Graph): The template graph
        world (Graph): The world graph
        candidates (np.array): The candidate matrix
        unspec_cover (np.array): Array of the indices of the nodes with
            more than 1 candidate
        verbose (bool): Verbosity flag
        init_changed_cands (np.array): A binary array where element i is 1
            if vertex i's candidates have changed since the function was last
            called. The first time it is called, this will be all zeros
        tmplt_equivalence (bool): Flag indicating whether to use template
            equivalence
        world_equivalence (bool): Flag indicating whether to use world
            equivalence
    Returns:
        int: The number of isomorphisms
    """
    # If the node cover is empty, the unspec nodes are disconnected. Thus, we
    # can skip straight to counting solutions to the alldiff constraint problem
    if len(unspec_cover) == 0:
        # Elimination filter is not needed here and would be a waste of time
        tmplt, world, candidates = run_filters(
                tmplt, world, candidates=candidates, filters=cheap_filters,
                    verbose=False, init_changed_cands=init_changed_cands)
        node_to_cands = {node: world.nodes[candidates[idx]]
                         for idx, node in enumerate(tmplt.nodes)}
        return count_alldiffs(node_to_cands)

    tmplt, world, candidates = run_filters(tmplt, world, candidates=candidates,
                filters=all_filters, verbose=False, 
                init_changed_cands=init_changed_cands)

    # Since the node cover is not empty, we first choose some valid
    # assignment of the unspecified nodes one at a time until the remaining
    # unspecified nodes are disconnected.
    n_isomorphisms = 0
    unspec_cover_cands = candidates[unspec_cover,:]
    node_idx = pick_minimum_vertex(unspec_cover_cands)
    cand_idxs = np.argwhere(candidates[node_idx]).flat

    for i, cand_idx in enumerate(cand_idxs):
        candidates_copy = candidates.copy()
        candidates_copy[node_idx] = one_hot(cand_idx, world.n_nodes)

        # Remove matched node from the unspecified list
        new_unspec_cover = unspec_cover[:node_idx] + unspec_cover[node_idx+1:]

        # recurse to make assignment for the next node in the unspecified cover
        n_isomorphisms += recursive_isomorphism_counter(
            tmplt, world, candidates_copy, unspec_cover=new_unspec_cover,
            verbose=verbose, init_changed_cands=one_hot(node_idx, tmplt.n_nodes))

        # TODO: more useful progress summary
        if verbose:
            print("depth {}: {} of {}".format(len(unspec_cover), i, 
                                              len(cand_idxs)), n_isomorphisms)

    return n_isomorphisms


def count_isomorphisms(tmplt, world, *, candidates=None, verbose=True,
                       tmplt_equivalence=False, world_equivalence=False):
    """
    counts the number of ways to assign template nodes to world nodes such that
    edges between template nodes also appear between the corresponding world
    nodes. Does not factor in the number of ways to assign the edges. Only
    counts the number of assignments between nodes.

    if the set of unspecified template nodes is too large or too densely
    connected, this code may never finish.

    Args:
        tmplt (Graph): The template graph
        world (Graph): The world graph
        candidates (np.array): The candidate matrix
        verbose (bool): Verbosity flag
        tmplt_equivalence (bool): Flag indicating whether to use template
            equivalence
        world_equivalence (bool): Flag indicating whether to use world
            equivalence
    Returns:
        int: The number of isomorphisms
    """

    if candidates is None:
        tmplt, world, candidates = uclasm.run_filters(
            tmplt, world, filters=uclasm.all_filters, verbose=verbose)

    unspec_nodes = np.where(candidates.sum(axis=1) > 1)[0]
    tmplt_subgraph = tmplt.subgraph(unspec_nodes)
    unspec_cover = get_node_cover(tmplt_subgraph)
    unspec_cover_nodes = [tmplt_subgraph.nodes[node_idx] for node_idx in unspec_cover]
    unspec_cover_idxes = [tmplt.node_idxs[node] for node in unspec_cover_nodes]

    # Send zeros to init_changed_cands since we already just ran the filters
    return recursive_isomorphism_counter(
        tmplt, world, candidates, verbose=verbose, unspec_cover=unspec_cover_idxes,
        init_changed_cands=np.zeros(tmplt.nodes.shape, dtype=np.bool))
