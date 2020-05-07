"""Provides a method for performing a greedy search for solutions using
total costs."""

import numpy as np

from .search_utils import *
from ..global_cost_bound import *
from ..local_cost_bound import *
from ...utils import one_hot
from heapq import heappush, heappop

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
    start_state.cost = smp.global_costs.min()
    cost_map[start_state.matching] = start_state.cost

    # Handle the case where we start in a solved state
    if len(start_state.matching) == smp.tmplt.n_nodes:
        solutions.append(start_state)
        return solutions

    heappush(open_list, start_state)

    # Maximum cost for a matching to be considered
    max_cost = smp.global_cost_threshold
    # Cost of the kth solution
    kth_cost = float("inf")

    while len(open_list) > 0:
        curr_smp = smp.copy()
        current_state = heappop(open_list)
        if verbose:
            print("Current state: {} matches".format(len(current_state.matching)),
                  "{} open states".format(len(open_list)), "current_cost:", current_state.cost,
                  "kth_cost:", kth_cost, "solutions found:", len(solutions))
        # Ignore states whose cost is too high
        if current_state.cost > max_cost or current_state.cost >= kth_cost:
            continue
        set_fixed_costs(curr_smp.fixed_costs, current_state.matching)
        curr_smp.set_costs(local_costs=np.zeros(curr_smp.shape),
                           global_costs=np.zeros(curr_smp.shape))
        # Do not reduce world as it can mess up the world indices in the matching
        iterate_to_convergence(curr_smp, reduce_world=False)
        global_costs = curr_smp.global_costs
        matching_dict = dict_from_tuple(current_state.matching)
        candidates = np.logical_and(global_costs <= max_cost, global_costs < kth_cost)
        # Identify template node with the least number of candidates
        cand_counts = np.sum(candidates, axis=1)
        # Prevent previously matched template idxs from being chosen
        cand_counts[list(matching_dict)] = np.max(cand_counts) + 1
        tmplt_idx = np.argmin(cand_counts)
        cand_idxs = np.argwhere(candidates[tmplt_idx])
        if verbose:
            print("Choosing candidate for", tmplt_idx,
                  "with {} possibilities".format(len(cand_idxs)))

        # Only push states that have a total cost bound lower than the threshold
        for cand_idx in cand_idxs:
            cand_idx = cand_idx[0]
            new_matching = matching_dict.copy()
            new_matching[tmplt_idx] = cand_idx
            new_matching_tuple = tuple_from_dict(new_matching)
            if new_matching_tuple not in cost_map:
                new_state = State()
                new_state.matching = new_matching_tuple
                # temp_smp = curr_smp.copy()
                # set_fixed_costs(temp_smp.fixed_costs, new_state.matching)
                # # Reset the costs to account for potential increase
                # temp_smp.set_costs(local_costs=np.zeros(temp_smp.shape),
                #                    global_costs=np.zeros(temp_smp.shape))
                # # Do not reduce world as it can mess up the world indices in the matching
                # iterate_to_convergence(temp_smp, reduce_world=False)
                # new_state.cost = temp_smp.global_costs.min()
                new_state.cost = global_costs[tmplt_idx, cand_idx]
                if new_state.cost > max_cost or new_state.cost >= kth_cost:
                    continue
                cost_map[new_matching_tuple] = new_state.cost
                if len(new_state.matching) == smp.tmplt.n_nodes:
                    solutions.append(new_state)
                    if k > 0 and len(solutions) > k:
                        solutions.sort()
                        solutions.pop()
                        kth_cost = min(solutions).cost
                else:
                    heappush(open_list, new_state)
            else:
                if verbose:
                    print("Recognized state: ", new_matching)
    if verbose:
        for solution in solutions:
            print(solution)
    return solutions
