import sys
sys.path.append("..")
import uclasm

import numpy as np
from functools import reduce

n_iterations = 0

def validate_alldiff_solns(tmplt, world, candidates):
    """ Check that there exists a solution to the alldiff problem """
    # Map from tmplt index to possible candidate indices
    var_to_vals = {
        tmplt_idx: [
            cand_idx for cand_idx in range(world.n_nodes)
            if candidates[tmplt_idx, cand_idx]
        ]
        for tmplt_idx in range(tmplt.n_nodes)
    }

    # if a var has only one possible val, track it then throw it out.
    matched_pairs = [(var, list(vals)[0]) for var, vals in var_to_vals.items()
                     if len(vals) == 1] # TODO: better variable name
    var_to_vals = {var: vals for var, vals in var_to_vals.items()
                   if len(vals) > 1}

    unspec_vars = list(var_to_vals.keys())
    # which vars is each val a cand for?
    val_to_vars = uclasm.invert(var_to_vals)

    # gather sets of vals which have the same set of possible vars.
    vars_to_vals = uclasm.values_map_to_same_key(val_to_vars)
    vars_to_val_counts = {vars: len(vals)
              for vars, vals in vars_to_vals.items()}

    # each var can belong to multiple sets of vars which key vars_to_val_counts
    # so here we find out which sets of vars each var belongs to
    var_to_vars_list = {
        var: [vars for vars in vars_to_val_counts.keys() if var in vars]
        for var in var_to_vals}

    def recursive_validate(var_to_vars_list, vars_to_vals, vars_to_val_counts):
        global n_iterations
        n_iterations += 1
        if len(var_to_vars_list) == 0:
            return True
        # Retrieve an arbitrary unspecified variable
        var, vars_list = var_to_vars_list.popitem()
        # Iterate through possible assignments of that variable
        for vars in vars_list:
            # How many ways are there to assign the variable in this way?
            n_vals = vars_to_val_counts[vars]
            if n_vals == 0:
                continue
            vars_to_val_counts[vars] -= 1
            if recursive_validate(var_to_vars_list, vars_to_vals, vars_to_val_counts):
                return True
            # put the count back so we don't mess up the recursion
            vars_to_val_counts[vars] += 1
        # put the list back so we don't mess up the recursion
        var_to_vars_list[var] = vars_list
        return False

    return recursive_validate(var_to_vars_list, vars_to_vals, vars_to_val_counts)

# TODO: switch to keyword arguments throughout
def validate_isomorphisms(tmplt, world, candidates, unspec_cover):
    """ Validate that at least one isomorphism exists"""
    global n_iterations
    n_iterations += 1
    if len(unspec_cover) == 0:
        return validate_alldiff_solns(tmplt, world, candidates)

    unspec_idx = unspec_cover[0]
    unspec_cands = np.argwhere(candidates[unspec_idx]).flat

    for cand_idx in unspec_cands:
        # Make a copy to avoid messing up candidate sets during recursion
        candidates_copy = candidates.copy()
        candidates_copy[unspec_idx, :] = uclasm.one_hot(cand_idx, world.n_nodes)

        # rerun filters after picking an assignment for the next unspec node
        _, new_world, new_candidates = uclasm.run_filters(
            tmplt, world, candidates=candidates_copy,
            filters=uclasm.cheap_filters,
            init_changed_cands=uclasm.one_hot(unspec_idx, tmplt.n_nodes))

        # if any node has no cands due to the current assignment, skip
        if not new_candidates.any(axis=1).all():
            continue

        if validate_isomorphisms(tmplt, new_world, new_candidates,
                                 unspec_cover[1:]):
            return True
    return False

def has_isomorphism(tmplt, world, *, candidates=None, verbose=False, count_iterations=False, **kwargs):
    """
    Searches for an isomorphism and returns true if one is found, else returns false
    """
    global n_iterations
    n_iterations = 0
    if candidates is None:
        tmplt, world, candidates = uclasm.run_filters(
            tmplt, world, filters=uclasm.all_filters,
            candidates=np.ones((tmplt.n_nodes, world.n_nodes), dtype=np.bool),
            **kwargs)

    # TODO: only recompute unspec_cover when necessary or not at all
    # Get node cover for unspecified nodes
    cand_counts = candidates.sum(axis=1)
    unspec_subgraph = tmplt.subgraph(cand_counts > 1)
    unspec_cover = uclasm.get_node_cover(unspec_subgraph)
    unspec_cover = np.array([tmplt.node_idxs[unspec_subgraph.nodes[idx]]
                             for idx in unspec_cover], dtype=np.int)

    # TODO: pass arguments as keywords to avoid bugs when changes are made
    if validate_isomorphisms(tmplt, world, candidates, unspec_cover):
        if count_iterations:
            return True, n_iterations
        return True
    if count_iterations:
        return False, n_iterations
    return False
