import uclasm

import numpy as np
from functools import reduce

# TODO: switch to keyword arguments throughout
def validate_alldiff_solns(tmplt, world, candidates, marked,
                           in_signal_only, node_to_marked_col_idx):
    """ Check that there exists a solution to the alldiff problem """
    # Map from tmplt index to possible candidate indices
    var_to_vals = {
        tmplt_idx: [
            node_to_marked_col_idx[world.nodes[cand_idx]]
            for cand_idx in range(world.n_nodes)
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
        if len(var_to_vars_list) == 0:
            return True
        # Retrieve an arbitrary unspecified variable
        var, vars_list = var_to_vars_list.popitem()
        found = False
        # Iterate through possible assignments of that variable
        for vars in vars_list:
            # How many ways are there to assign the variable in this way?
            n_vals = vars_to_val_counts[vars]
            if n_vals == 0:
                continue
            vars_to_val_counts[vars] -= 1
            if recursive_validate(var_to_vars_list, vars_to_vals, vars_to_val_counts):
                found = True
                # Unmark all nodes found
                marked[np.ix_(list(vars), list(vars_to_vals[vars]))] = False
            # put the count back so we don't mess up the recursion
            vars_to_val_counts[vars] += 1
        # put the list back so we don't mess up the recursion
        var_to_vars_list[var] = vars_list
        return found

    if recursive_validate(var_to_vars_list, vars_to_vals, vars_to_val_counts):
        # Unmark all pairs that were matched at the beginning
        for matched_pair in matched_pairs:
            if in_signal_only:
                # Unmark all pairs corresponding to the found candidate
                marked[:, matched_pair[1]] = False
            else:
                marked[matched_pair] = False
        return True
    return False

# TODO: switch to keyword arguments throughout
def validate_isomorphisms(tmplt, world, candidates, unspec_cover, marked,
                          in_signal_only, node_to_marked_col_idx):
    """ Validate that at least one isomorphism exists and unmark it """
    if len(unspec_cover) == 0:
        return validate_alldiff_solns(tmplt, world, candidates, marked,
                                      in_signal_only, node_to_marked_col_idx)

    unspec_idx = unspec_cover[0]
    unspec_cands = np.argwhere(candidates[unspec_idx]).flat

    # TODO: is this actually an effective heuristic? Compare with random order
    # Order unspec_cands to have marked nodes first
    unspec_cands = sorted(unspec_cands,
                          key=lambda cand_idx: marked[unspec_idx, cand_idx],
                          reverse=True)

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
                                 unspec_cover[1:], marked, in_signal_only,
                                 node_to_marked_col_idx):
            marked_col_idx = node_to_marked_col_idx[world.nodes[cand_idx]]
            if in_signal_only:
                # Unmark all pairs for the found candidate
                marked[:, marked_col_idx] = False
            else:
                # Unmark the found pair
                marked[unspec_idx, marked_col_idx] = False
            return True
    return False

def validation_filter(tmplt, world, *, candidates=None, in_signal_only=False,
                      verbose=False, **kwargs):
    """
    This filter finds the minimum candidate set for each template node by
    identifying one isomorphism for each candidate-template node pair

    in_signal_only: Rather than checking pairs, if this option is True, only
     check that each candidate participates in at least one signal, ignoring
     which template node it corresponds to
    """
    if candidates is None:
        tmplt, world, candidates = uclasm.run_filters(
            tmplt, world, filters=uclasm.all_filters,
            candidates=np.ones((tmplt.n_nodes, world.n_nodes), dtype=np.bool),
            **kwargs)

    # Start by marking every current candidate-template node pair to be checked
    # A zero entry here means that we have already checked whether or not the
    # candidate corresponds to the template node in any signals.
    marked = candidates.copy()

    node_to_marked_col_idx = {node: idx for idx, node in enumerate(world.nodes)}

    while marked.any():
        if verbose:
            print(marked.sum(), "marks remaining")

        candidates_copy = candidates.copy()

        # TODO: only recompute unspec_cover when necessary or not at all
        # Get node cover for unspecified nodes
        cand_counts = candidates.sum(axis=1)
        unspec_subgraph = tmplt.subgraph(cand_counts > 1)
        unspec_cover = uclasm.get_node_cover(unspec_subgraph)
        unspec_cover = np.array([tmplt.node_idxs[unspec_subgraph.nodes[idx]]
                                 for idx in unspec_cover], dtype=np.int)

        # Find a marked template node idx and a cand to pair together

        # Pick any pair with a mark
        marked_tmplt_idx, marked_cand_idx = np.argwhere(marked)[0]

        # unspecified template nodes which have any marks
        marked_unspecs = marked[unspec_cover].any(axis=1)

        # If there is a node in the unspec cover with a mark, prioritize it
        if marked_unspecs.any():
            # Pick the first node in the unspec cover that has a mark
            marked_tmplt_idx = unspec_cover[marked_unspecs][0]

            # Set a candidate for the marked template node as the marked cand
            marked_cand_idx = np.argwhere(marked[marked_tmplt_idx])[0,0]

        candidates_copy[marked_tmplt_idx, :] = uclasm.one_hot(marked_cand_idx,
                                                              world.n_nodes)

        # TODO: pass arguments as keywords to avoid bugs when changes are made
        if not validate_isomorphisms(tmplt, world, candidates_copy,
                                     unspec_cover, marked, in_signal_only,
                                     node_to_marked_col_idx):
            # No valid isomorphisms: remove from is_cand
            candidates[marked_tmplt_idx, marked_cand_idx] = False
            # Unmark the pair that was checked
            marked[marked_tmplt_idx, marked_cand_idx] = False
            # TODO: run cheap filters to propagate change of candidates
            # TODO: reduce world to cands
        elif in_signal_only:
            # Unmark all pairs for the candidate that was found
            marked[:, marked_cand_idx] = False
        else:
            # Unmark the pair that was found
            marked[marked_tmplt_idx, marked_cand_idx] = False

    return tmplt, world, candidates
