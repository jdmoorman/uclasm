"""Provides a method for performing a greedy search for solutions using
total costs."""

import numpy as np

from ..global_cost_bound import *
from ..local_cost_bound import *
from heapq import heappush, heappop

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

def greedy_best_k_matching(smp, k=1, verbose=False):
    """Greedy search on the cost heuristic to find the best k matchings.
    Parameters
    ----------
    smp: MatchingProblem
        A subgraph matching problem.
    k: int
        The maximum number of solutions to find. Note that the cost thresholds
        in smp may result in this function returning fewer than n solutions, if
        there are not enough solutions satisfying the cost thresholds.
    """
    # States still left to be processed
    open_list = []
    # States which have already been processed
    closed_list = []
    # Map from states to computed costs
    cost_map = {}
    # States where all nodes have been assigned
    solutions = []

    # Map from template indexes to world indexes
    current_matching = {}

    # Initialize matching with known matches
    candidates = smp.candidates()
    for i in range(smp.tmplt.n_nodes):
        if np.sum(candidates[i]) == 1:
            current_matching[i] = np.argwhere(candidates[i])[0][0]

    start_state = State()
    start_state.matching = tuple_from_dict(current_matching)
    start_state.cost = smp.total_costs.min()
    cost_map[start_state.matching] = start_state.cost

    # Handle the case where we start in a solved state
    if len(start_state.matching) == smp.tmplt.n_nodes:
        solutions.append(start_state)
        return solutions

    heappush(open_list, start_state)

    # Maximum cost for a matching to be considered
    max_cost = smp.global_cost_threshold

    while len(open_list) > 0:
        curr_smp = smp.copy()
        current_state = heappop(open_list)
        set_fixed_costs(curr_smp.fixed_costs, current_state.matching)
        nodewise(curr_smp)
        edgewise(curr_smp)
        from_local_bounds(curr_smp)
        global_costs = curr_smp.global_costs
        matching_dict = dict_from_tuple(current_state.matching)
        # Only push states that have a total cost bound lower than the threshold
        for tmplt_idx, cand_idx in np.argwhere(global_costs < max_cost):
            if tmplt_idx not in matching_dict:
                new_matching = matching_dict.copy()
                new_matching[tmplt_idx] = cand_idx
                new_matching_tuple = tuple_from_dict(new_matching)
                if new_matching_tuple not in cost_map:
                    new_state = State()
                    new_state.matching = new_matching_tuple
                    temp_smp = smp.copy()
                    set_fixed_costs(temp_smp.fixed_costs, current_state.matching)
                    nodewise_cost_bound(temp_smp)
                    edgewise_cost_bound(temp_smp)
                    from_local_bounds(temp_smp)
                    new_state.cost = temp_smp.global_costs.min()
                    cost_map[new_matching_tuple] = new_state.cost
                    if len(new_state.matching) == smp.tmplt.n_nodes:
                        solutions.append(new_state)
                        if len(solutions) > k:
                            solutions.sort()
                            max_cost = solutions[-2].cost
                            solutions.pop()
                    else:
                        heappush(open_list, new_state)
    if verbose:
        for solution in solutions:
            print(solution)
    return solutions
