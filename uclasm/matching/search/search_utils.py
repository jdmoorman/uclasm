"""Utility functions and classes for search"""
import numpy as np

from .. import global_cost_bound
from .. import local_cost_bound

class State:
    """A state for the greedy search algorithm.
    Attributes
    ----------
    matching : tuple
        Tuple representation of the current matching.
    cost: float
        Estimated cost of the matching.
    """
    def __init__(self):
        self.matching = None
        self.cost = float("inf")

    def __lt__(self, other):
        if len(self.matching) != len(other.matching):
            return len(self.matching) > len(other.matching)
        return self.cost < other.cost

    def __str__(self):
        return str(self.matching) + ": " + str(self.cost)

def tuple_from_dict(dict):
    """Turns a dict into a representative sorted tuple of 2-tuples.
    Parameters
    ----------
    dict: dict
        A dictionary.
    Returns
    -------
    tuple
        A unique tuple representation of the dictionary as a sequence of
        2-tuples of key-value pairs, sorted by the dictionary key.
    """
    return tuple(sorted(dict.items()))

def dict_from_tuple(tuple):
    """Turns a tuple of 2-tuples into a dict.
    Parameters
    ----------
    tuple: tuple
        A unique tuple representation of the dictionary as a sequence of
        2-tuples of key-value pairs. Keys must be unique.
    Returns
    -------
    dict
        The equivalent dictionary.
    """
    return dict(tuple)

def set_fixed_costs(fixed_costs, matching):
    """Set fixed costs to float("inf") to enforce the given matching."""
    mask = np.zeros(fixed_costs.shape, dtype=np.bool)
    mask[[pair[0] for pair in matching],:] = True
    mask[:,[pair[1] for pair in matching]] = True
    mask[tuple(np.array(matching).T)] = False
    fixed_costs[mask] = float("inf")

def iterate_to_convergence(smp, verbose=False):
    changed_cands = np.ones((smp.tmplt.n_nodes,), dtype=np.bool)
    smp.global_costs = smp.local_costs/2 + smp.fixed_costs
    if verbose:
        print(smp)
        print("Running nodewise cost bound")
    local_cost_bound.nodewise(smp)
    global_cost_bound.from_local_bounds(smp)
    while True:
        old_candidates = smp.candidates().copy()
        if verbose:
            print(smp)
            print("Running edgewise cost bound")
        local_cost_bound.edgewise(smp)
        global_cost_bound.from_local_bounds(smp)
        changed_cands = np.any(smp.candidates() != old_candidates, axis=1)
        if ~np.any(changed_cands):
            break
    if verbose:
        print(smp)
