from ..utils import invert, values_map_to_same_key
import numpy as np
from functools import reduce

def recursive_alldiff_counter(tnode_to_eqids, eq_class_sizes):
    # If no more tnodes to assign
    if len(tnode_to_eqids) == 0:
        return 1

    count = 0

    # This tnode will now be assigned to one of the eq_classes
    tnode, eqids = tnode_to_eqids.popitem()

    # We try mapping the tnode to each equivalence class of world nodes
    for eqid in eqids:
        # TODO: break out this inner loop into its own function for clarity?
        n_cands = eq_class_sizes[eqid]

        # If no candidates left in this equivalence class, continue
        if n_cands == 0:
            continue

        # We decrement the amount of available candidates in the eq class.
        eq_class_sizes[eqid] -= 1

        # Count the number of ways to assign the rest of candidates
        n_ways_to_assign_rest = recursive_alldiff_counter(tnode_to_eqids, eq_class_sizes)

        # We have n_cands possible ways to assign the current tnode to the
        # current equivalence class, so we multiply by n_cands
        count += n_cands * n_ways_to_assign_rest

        # Unmatch tnode from equivalence class
        eq_class_sizes[eqid] += 1

    tnode_to_eqids[tnode] = eqids

    return count


def get_equivalence_classes(tnode_to_cands):
    """Get equivalence classes of cands which are candidates for the same nodes.

    Parameters
    ----------
    tnode_to_cands : dict
        Mapping from each template node to its candidates.

    Returns
    -------
    tnode_to_eqids : dict
        Mapping from template node to the equivalence classes to which its candidates belong.
    eq_classes : list
        List of the equivalence classes
    """

    tnodes = tnode_to_cands.keys()
    all_cands = set().union(*tnode_to_cands.values())

    # which set of template nodes is each cand a candidate for?
    cand_to_tnode_set = invert(tnode_to_cands)

    # gather sets of cands which have the same set of possible template nodes.
    # Each cand_set here is an equivalence class
    tnode_set_to_eq_class = values_map_to_same_key(cand_to_tnode_set)
    eq_classes = list(tnode_set_to_eq_class.values())

    cand_to_eqids = {
        cand: [eqid for eqid, eq_class in enumerate(eq_classes) if cand in eq_class]
        for cand in all_cands}

    # This is a dictionary mapping a tnode to the list of indices of
    # equivalence classes the tnode can map into
    tnode_to_eqids = {
        tnode: [
            eq_classes.index(tnode_set_to_eq_class[tnode_set])
            for tnode_set in tnode_set_to_eq_class
            if tnode in tnode_set
        ]
        for tnode in tnodes
    }

    return tnode_to_eqids, eq_classes

def count_alldiffs(tnode_to_cands):
    """
    tnode_to_cands: dict(item, list)

    Count the number of ways to assign template nodes to candidates without
    using any candidate for more than one template node.
    i.e. count solutions to the alldiff problem, where the nodeiables
    are the keys of tnode_to_cands, and the domains are the values.
    """

    # TODO: can this function be vectorized?
    # TODO: does scipy have a solver for this already?

    # Check if any template node has no candidates
    if any(len(cands)==0 for cands in tnode_to_cands.values()):
        return 0
    tnode_to_eqids, eq_classes = get_equivalence_classes(tnode_to_cands)

    # Alternatively, list(map(len, eq_classes))
    eq_class_sizes = [len(eq_class) for eq_class in eq_classes]

    count = recursive_alldiff_counter(tnode_to_eqids, eq_class_sizes)

    return count
