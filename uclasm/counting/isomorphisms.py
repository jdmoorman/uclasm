from ..matching.search.search_utils import iterate_to_convergence
from ..utils import invert, values_map_to_same_key, one_hot
from .alldiffs import count_alldiffs
import numpy as np
import pandas as pd
from functools import reduce

# TODO: count how many isomorphisms each background node participates in.
# TODO: switch from recursive to iterative implementation for readability

def pick_minimum_domain_vertex(candidates):
    """
    This function will return the index of the template vertex with the
    smallest domain that has at least 2 candidates. If there are multiple
    minima, pick the one with the smallest index.

    Parameters
    ----------
    candidates : np.array
        The candidate matrix
    Returns
    -------
    int
        The index of the template vertex
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

def recursive_isomorphism_counter(smp, matching, *,
        unspec_cover, verbose, init_changed_cands, tmplt_equivalence=False,
        world_equivalence=False):
    """
    Recursive routine for solving subgraph isomorphism.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem
    matching : list
        A list of tuples which designate what each template vertex is matched to
    unspec_cover : np.array
        Array of the indices of the nodes with more than 1 candidate
    verbose : bool
        Flag for verbose output
    init_changed_cands : np.array
        A binary array where element i is 1 if vertex i's candidates have
        changed since the function was last called. The first time it is called,
        this will be all zeros
    tmplt_equivalence : bool
        Flag indicating whether to use template equivalence
    world_equivalence : bool
        Flag indicating whether to use world equivalence
    Returns
    -------
    int
        The number of isomorphisms
    """

    iterate_to_convergence(smp)
    candidates = smp.candidates()

    # If the node cover is empty, the unspec nodes are disconnected. Thus, we
    # can skip straight to counting solutions to the alldiff constraint problem
    if len(unspec_cover) == 0:
        # Elimination filter is not needed here and would be a waste of time
        node_to_cands = {node: smp.world.nodes[candidates[idx]]
                         for idx, node in enumerate(smp.tmplt.nodes)}
        return count_alldiffs(node_to_cands)

    # Since the node cover is not empty, we first choose some valid
    # assignment of the unspecified nodes one at a time until the remaining
    # unspecified nodes are disconnected.
    n_isomorphisms = 0
    unspec_cover_cands = candidates[unspec_cover,:]
    node_idx = pick_minimum_domain_vertex(unspec_cover_cands)
    cand_idxs = np.argwhere(candidates[node_idx]).flat

    for i, cand_idx in enumerate(cand_idxs):
        smp_copy = smp.copy()
        # candidates_copy[node_idx] = one_hot(cand_idx, world.n_nodes)
        smp_copy.add_match(node_idx, cand_idx)

        matching.append((node_idx, cand_idx))
        # Remove matched node from the unspecified list
        new_unspec_cover = unspec_cover[:node_idx] + unspec_cover[node_idx+1:]

        # recurse to make assignment for the next node in the unspecified cover
        n_isomorphisms += recursive_isomorphism_counter(
            smp_copy, matching, unspec_cover=new_unspec_cover,
            verbose=verbose,
            init_changed_cands=one_hot(node_idx, smp.tmplt.n_nodes))

        # Unmatch template vertex
        matching.pop()

        # TODO: more useful progress summary
        if verbose:
            print("depth {}: {} of {}".format(len(unspec_cover), i,
                                              len(cand_idxs)), n_isomorphisms)

        # If we are using template equivalence, we can mark for all equivalent
        # template vertices that cand_idx cannot be a cannot be a candidate.
        if tmplt_equivalence:
            for eq_t_vert in smp.tmplt.eq_classes[node_idx]:
                smp.prevent_match(eq_t_vert, cand_idx)

    return n_isomorphisms


def count_isomorphisms(smp, *, verbose=True,
                       tmplt_equivalence=False, world_equivalence=False):
    """
    Counts the number of ways to assign template nodes to world nodes such that
    edges between template nodes also appear between the corresponding world
    nodes. Does not factor in the number of ways to assign the edges. Only
    counts the number of assignments between nodes.

    if the set of unspecified template nodes is too large or too densely
    connected, this code may never finish.

    Parameters
    ----------
    smp : Matching Problem
        A subgraph matching problem
    verbose : bool
        Flag for verbose output
    tmplt_equivalence : bool
        Flag indicating whether to use template equivalence
    world_equivalence : bool
        Flag indicating whether to use world equivalence
    Returns
    -------
    int
        The number of isomorphisms
    """

    matching = []
    candidates = smp.candidates()
    spec_nodes = np.where(candidates.sum(axis=1) == 1)[0]
    for t_vert in spec_nodes:
        w_vert = np.where(candidates[t_vert,:])[0][0]
        matching.append((t_vert, w_vert))

    unspec_nodes = np.where(candidates.sum(axis=1) > 1)[0]
    tmplt_subgraph = smp.tmplt.node_subgraph(unspec_nodes)
    unspec_cover_subgraph_idxs = tmplt_subgraph.node_cover()
    # Remap indices from subgraph back to original template
    unspec_cover_nodes = tmplt_subgraph.nodes[unspec_cover_subgraph_idxs]
    unspec_cover_idxs = [smp.tmplt.node_idxs[node] for node in unspec_cover_nodes]

    # Send zeros to init_changed_cands since we already just ran the filters
    return recursive_isomorphism_counter(
        smp, matching, verbose=verbose, unspec_cover=unspec_cover_idxs,
        init_changed_cands=np.zeros(smp.tmplt.nodes.shape, dtype=np.bool))

def recursive_isomorphism_finder(smp, *,
                                 unspec_node_idxs, verbose, init_changed_cands,
                                 found_isomorphisms):
    if len(unspec_node_idxs) == 0:
        # All nodes have been assigned, add the isomorphism to the list
        new_isomorphism = {}
        candidates = smp.candidates()
        for tmplt_idx, tmplt_node in enumerate(smp.tmplt.nodes):
            if verbose:
                world_node = smp.world.nodes[candidates[tmplt_idx]]
                if isinstance(world_node, pd.Series):
                    world_node = world_node.iloc[0]
                print(str(tmplt_node)+":", world_node)
                new_isomorphism[tmplt_node] = world_node
        found_isomorphisms.append(new_isomorphism)
        return found_isomorphisms

    iterate_to_convergence(smp)
    candidates = smp.candidates()

    node_idx = unspec_node_idxs[0]
    cand_idxs = np.argwhere(candidates[node_idx]).flat

    for i, cand_idx in enumerate(cand_idxs):
        smp_copy = smp.copy()
        smp.add_match(node_idx, cand_idx)

        # recurse to make assignment for the next node in the unspecified cover
        recursive_isomorphism_finder(
            smp_copy,
            unspec_node_idxs=unspec_node_idxs[1:],
            verbose=verbose,
            init_changed_cands=one_hot(node_idx, smp.tmplt.n_nodes),
            found_isomorphisms=found_isomorphisms)
    return found_isomorphisms

def find_isomorphisms(smp, *, verbose=True):
    """ Returns a list of isomorphisms as dictionaries mapping template nodes to
    world nodes. Note: this is much slower than counting, and should only be
    done for small numbers of isomorphisms and fully filtered candidate matrices
    """
    unspec_node_idxs = np.where(smp.candidates().sum(axis=1) > 1)[0]
    found_isomorphisms = []

    return recursive_isomorphism_finder(
        smp, verbose=verbose,
        unspec_node_idxs=unspec_node_idxs,
        init_changed_cands=np.zeros(smp.tmplt.nodes.shape, dtype=np.bool),
        found_isomorphisms=found_isomorphisms)

def print_isomorphisms(smp, *, verbose=True):
    """ Prints the list of isomorphisms """
    print(find_isomorphisms(smp, verbose=verbose))
