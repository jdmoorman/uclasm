from ..utils.misc import invert, values_map_to_same_key
import numpy as np
from functools import reduce


# counts the number of ways to assign a particular var, recursing to
# incorporate ways to assign the next var, and so on
def recursive_alldiff_counter(var_to_vars_list, vars_to_val_counts):
    # no vars left to assign
    if len(var_to_vars_list) == 0:
        return 1

    count = 0

    # give me an arbitrary unspecified variable
    var, vars_list = var_to_vars_list.popitem()

    # for each way of assigning the given variable
    for vars in vars_list:
        # how many ways to assign the variable in this way are there?
        n_vals = vars_to_val_counts[vars]
        if n_vals == 0:
            continue

        vars_to_val_counts[vars] -= 1

        # number of ways to assign current var times number of ways to
        # assign the rest
        n_ways_to_assign_rest = recursive_alldiff_counter(
            var_to_vars_list, vars_to_val_counts)
                                            
        count += n_vals * n_ways_to_assign_rest

        # put the count back so we don't mess up the recursion
        vars_to_val_counts[vars] += 1

    # put the list back so we don't mess up the recursion
    var_to_vars_list[var] = vars_list

    return count

def count_alldiffs(var_to_vals):
    """
    var_to_vals: dict(item, list)

    count the number of ways to assign vars to vals without using any val for
    more than one var. ie. count solns to alldiff problem, where the variables
    are the keys of var_to_vals, and the domains are the values.
    """
    
    # TODO: can this function be vectorized?
    # TODO: does scipy have a solver for this already?

    # Check if any var has no vals
    if any(len(vals)==0 for vals in var_to_vals.values()):
        return 0

    # TODO: throwing out vars with only one val may not be necessary
    # if a var has only one possible val, throw it out.
    var_to_vals = {var: vals for var, vals in var_to_vals.items()
                   if len(vals) > 1}

    unspec_vars = list(var_to_vals.keys())

    # which vars is each val a cand for?
    val_to_vars = invert(var_to_vals)

    # gather sets of vals which have the same set of possible vars.
    vars_to_vals = values_map_to_same_key(val_to_vars)
    vars_to_val_counts = {vars: len(vals)
                          for vars, vals in vars_to_vals.items()}

    # each var can belong to multiple sets of vars which key vars_to_val_counts
    # so here we find out which sets of vars each var belongs to
    var_to_vars_list = {
        var: [vars for vars in vars_to_val_counts.keys() if var in vars]
        for var in var_to_vals}

    return recursive_alldiff_counter(var_to_vars_list, vars_to_val_counts)
