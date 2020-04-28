"""
alldiffs.py

Routine for computing solutions to the all different problem
"""

from math import factorial

import numpy as np
from scipy.special import comb

from ..utils import invert, values_map_to_same_key
from ..equivalence import equivalence_from_partition

# Right now we are using a global variable to keep track of solution count
num_solutions = 0


def count_solutions(matching, smp, w_classes, tmplt_equivalence=False):
    """
    This function will compute the number of solutions for a given matching.

    Parameters
    ----------
    matching : dict
        The matching for which we want to compute solutions
    smp : MatchingProblem
        The matching problem context
    w_classes : list
        A list of sets of world nodes that are equivalent
    tmplt_equivalence : bool
        A flag indicating whether to use template equivalence

    Returns
    -------
    int : The number of solutions that can be generated using template and
        world equivalence
    """
    count = 1

    w_class_sizes = [len(w_class) for w_class in w_classes]
    if tmplt_equivalence:
        t_classes = smp.tmplt.equivalence_classes

        for t_class in t_classes.classes().values():
            # We can permute members of template equivalence classes freely
            # So we multiply by the factorial of the size of each class
            count *= factorial(t_class)
        
            # Then we need to determine within each template equivalence class
            # how many template nodes are assigned to the same world equivalence
            # class. Once we do this, we need to choose which elements of the
            # world equivalence class to assign to these template nodes.
            w_class_counts = {}
            for tnode in t_class:
                wnode = matching[tnode]
                # If class_id is None after this loop, then the tnode was
                # assigned prior to calling the alldiffs problem. It will be
                # assigned to only one node and not a class.
                class_id = None
                # Determine which equivalence class it is in
                for i, w_class in enumerate(w_classes):
                    if wnode in w_class:
                        class_id = i
                        break

                w_class_counts[class_id] = w_class_counts.get(class_id, 0) + 1
            
            for w_class_id in w_class_counts:
                if w_class_id is None:
                    continue

                # Choose which elements of the w_class to assign to the
                # members of the template class
                count *= comb(w_class_sizes[w_class_id],
                              w_class_counts[w_class_id], exact=True)

                w_class_sizes[w_class_id] -= w_class_counts[w_class_id]

    else:
        for tnode in matching:
            wnode = matching[tnode]
            class_id = None
            # Determine which equivalence class it is in
            for i, w_class in enumerate(w_classes):
                if wnode in w_class:
                    class_id = i
                    break
            if class_id is None:
                continue

            count *= w_class_sizes[class_id]
            w_class_sizes[class_id] -= 1

    return count


def recursive_alldiff_counter_SMP(tnode_to_eqids, w_classes, w_class_sizes,
                                  smp, matching, tmplt_equivalence=False):
    """
    Recursive routine for computing the number of ways to assign template
    nodes to world nodes such that they are all different. This version is
    intended for use with the subgraph matching problem.

    Parameters
    ----------
    tnode_to_eqids : dict
        Mapping from template node to a number representing a class of
        equivalence nodes
    w_classes : list
        A partition of world nodes into equivalence classes
    w_class_sizes : list
        A list of the number of currently unassigned world nodes for each
        equivalence class
    smp : SubgraphMatchingProblem
        The current matching problem
    matching : dict
        The current assignment of template nodes to world nodes, matching[i]
        is the index of the world node that template node i is assigned to
    tmplt_equivalence : bool
        A flag indicating whether we are using template equivalence
    """

    # We need to count the number of solutions the current matching represents
    if len(tnode_to_eqids) == 0:
        sol_count = count_solutions(matching, smp, w_classes, tmplt_equivalence)
        global num_solutions
        num_solutions += sol_count
        return
        

    # This tnode will now be assigned to one of the w_classes
    tnode, eqids = tnode_to_eqids.popitem()

    # We try mapping the tnode to each equivalence class of world nodes
    for eqid in eqids:
        # TODO: break out this inner loop into its own function for clarity?
        n_cands = w_class_sizes[eqid]

        # If no candidates left in this equivalence class, continue
        if n_cands == 0:
            continue

        # We decrement the amount of available candidates in the eq class.
        w_class_sizes[eqid] -= 1

        # Assign tnode to an arbitrary element of the equivalence class
        eq_class = list(w_classes[eqid])
        matching[tnode] = eq_class[w_class_sizes[eqid]]

        # Assign the rest
        recursive_alldiff_counter(tnode_to_eqids, w_classes, w_class_sizes,
                                  smp, matching, tmplt_equivalence)

        # Unmatch tnode
        matching[tnode] = -1
        # Restore the world node to the class its from
        w_class_sizes[eqid] += 1

    tnode_to_eqids[tnode] = eqids


def get_equivalence_classes(tnode_to_cands):
    """Get equivalence classes of cands which are candidates for the same 
    template nodes.

    Parameters
    ----------
    tnode_to_cands : dict
        Mapping from each template node to its candidates.

    Returns
    -------
    tnode_to_eqids : dict
        Mapping from template node to the equivalence classes to which its 
        candidates belong.
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
    w_equivalence = equivalence_from_partition(eq_classes)

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


def count_alldiffs_SMP(tnode_to_cands, smp, matching, tmplt_equivalence=False):
    """
    Count the number of ways to assign template nodes to candidates without
    using any candidate for more than one template node.
    i.e. count solutions to the alldiff problem, where the nodeiables
    are the keys of tnode_to_cands, and the domains are the values.

    Parameters
    ----------
    tnode_to_cands: dict(item, list)
    smp: SubgraphMatchingProblem
    matching: dict
    tmplt_equivalence: bool
    """
    global num_solutions
    num_solutions = 0

    # TODO: can this function be vectorized?
    # TODO: does scipy have a solver for this already?

    # Check if any template node has no candidates
    if any(len(cands)==0 for cands in tnode_to_cands.values()):
        return 0

    tnode_to_eqids, eq_classes = get_equivalence_classes(tnode_to_cands)

    # Alternatively, list(map(len, eq_classes))
    eq_class_sizes = [len(eq_class) for eq_class in eq_classes]

    recursive_alldiff_counter(tnode_to_eqids, eq_classes, 
                              eq_class_sizes, smp, matching, tmplt_equivalence)

    return num_solutions


def recursive_alldiff_counter(tnode_to_eqids, w_class_sizes):
    """
    Parameters
    ----------
    tnode_to_eqids : dict
        Mapping from template node to a number representing a class of
        equivalence nodes
    w_class_sizes : list
        A list of the number of currently unassigned world nodes for each
        equivalence class

    Returns
    -------
    The number of solutions to the all different problem
    """

    # If no more tnodes to assign
    if len(tnode_to_eqids) == 0:
        return 1

    count = 0

    # This tnode will now be assigned to one of the w_classes
    tnode, eqids = tnode_to_eqids.popitem()

    # We try mapping the tnode to each equivalence class of world nodes
    for eqid in eqids:
        # TODO: break out this inner loop into its own function for clarity?
        n_cands = w_class_sizes[eqid]

        # If no candidates left in this equivalence class, continue
        if n_cands == 0:
            continue

        # We decrement the amount of available candidates in the eq class.
        w_class_sizes[eqid] -= 1

        # Count the number of ways to assign the rest of candidates
        n_ways_to_assign_rest = recursive_alldiff_counter(tnode_to_eqids, w_class_sizes)

        # We have n_cands possible ways to assign the current tnode to the
        # current equivalence class, so we multiply by n_cands
        count += n_cands * n_ways_to_assign_rest

        # Unmatch tnode from equivalence class
        w_class_sizes[eqid] += 1

    tnode_to_eqids[tnode] = eqids

    return count


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
