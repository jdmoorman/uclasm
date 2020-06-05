"""Provides a method for performing a greedy search for solutions using
total costs."""

import numpy as np
import bisect

from .search_utils import *
from ..global_cost_bound import *
from ..local_cost_bound import *
from ..matching_problem import MatchingProblem
from ...utils import one_hot
from heapq import heappush, heappop, heapify

def greedy_best_k_matching(smp, k=1, nodewise=True, edgewise=True,
                           verbose=False):
    """Greedy search on the cost heuristic to find the best k matchings.
    Parameters
    ----------
    smp: MatchingProblem
        A subgraph matching problem.
    k: int
        The maximum number of solutions to find. Note that the cost thresholds
        in smp may result in this function returning fewer than n solutions, if
        there are not enough solutions satisfying the cost thresholds.
    nodewise: bool
        Whether to use the nodewise cost bound.
    edgewise: bool
        Whether to use the edgewise cost bound.
    """
    if smp.global_cost_threshold == float("inf"):
        raise Exception("Invalid global cost threshold.")

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
        current_state = heappop(open_list)
        # Ignore states whose cost is too high
        if current_state.cost > max_cost or current_state.cost >= kth_cost:
            # Only print multiples of 10000 for skipped states
            if verbose and len(open_list) % 10000 == 0:
                print("Skipped state: {} matches".format(len(current_state.matching)),
                      "{} open states".format(len(open_list)), "current_cost:", current_state.cost,
                      "kth_cost:", kth_cost, "max_cost", smp.global_cost_threshold, "solutions found:", len(solutions))
            continue
        if verbose:
            print("Current state: {} matches".format(len(current_state.matching)),
                  "{} open states".format(len(open_list)), "current_cost:", current_state.cost,
                  "kth_cost:", kth_cost, "max_cost", smp.global_cost_threshold, "solutions found:", len(solutions))

        curr_smp = smp.copy()
        set_fixed_costs(curr_smp.fixed_costs, current_state.matching)
        # global costs can increase as a result of candidate assignment
        curr_smp.set_costs(global_costs=np.zeros(curr_smp.shape))
        # Do not reduce world as it can mess up the world indices in the matching
        iterate_to_convergence(curr_smp, reduce_world=False, nodewise=nodewise,
                               edgewise=edgewise)
        global_costs = curr_smp.global_costs
        matching_dict = dict_from_tuple(current_state.matching)
        max_cost = curr_smp.global_cost_threshold
        candidates = np.logical_and(global_costs <= max_cost, global_costs < kth_cost)
        # Identify template node with the least number of candidates
        cand_counts = np.sum(candidates, axis=1)
        # Prevent previously matched template idxs from being chosen
        cand_counts[list(matching_dict)] = np.max(cand_counts) + 1
        tmplt_idx = np.argmin(cand_counts)
        cand_idxs = np.argwhere(candidates[tmplt_idx]).flatten()
        if verbose:
            print("Choosing candidate for", tmplt_idx,
                  "with {} possibilities".format(len(cand_idxs)))

        # Only push states that have a total cost bound lower than the threshold
        for cand_idx in cand_idxs:
            new_matching = matching_dict.copy()
            new_matching[tmplt_idx] = cand_idx
            new_matching_tuple = tuple_from_dict(new_matching)
            if new_matching_tuple not in cost_map:
                new_state = State()
                new_state.matching = new_matching_tuple
                new_state.cost = global_costs[tmplt_idx, cand_idx]
                if new_state.cost > max_cost or new_state.cost >= kth_cost:
                    continue
                cost_map[new_matching_tuple] = new_state.cost
                if len(new_state.matching) == smp.tmplt.n_nodes:
                    # temp_smp = curr_smp.copy()
                    # set_fixed_costs(temp_smp.fixed_costs, new_state.matching)
                    # # Reset the costs to account for potential increase
                    # temp_smp.set_costs(local_costs=np.zeros(temp_smp.shape),
                    #                    global_costs=np.zeros(temp_smp.shape))
                    # # Do not reduce world as it can mess up the world indices in the matching
                    # iterate_to_convergence(temp_smp, reduce_world=False,
                    #                        nodewise=nodewise, edgewise=edgewise)
                    # new_state.cost = temp_smp.global_costs.min()
                    solutions.append(new_state)
                    if k > 0 and len(solutions) > k:
                        solutions.sort()
                        solutions.pop()
                        heapify(solutions)
                        kth_cost = max(solutions).cost
                        smp.global_cost_threshold = min(smp.global_cost_threshold,
                                                        kth_cost)
                        iterate_to_convergence(smp, reduce_world=False, nodewise=nodewise,
                                                   edgewise=edgewise)
                else:
                    heappush(open_list, new_state)
            else:
                if verbose:
                    print("Recognized state: ", new_matching)
    if verbose and len(solutions) < 100:
        for solution in solutions:
            print(solution)
    return solutions

def copy_smp_shallow_graphs(smp):
    return MatchingProblem(smp.tmplt, smp.world,
        fixed_costs=smp._fixed_costs.copy(),
        local_costs=smp._local_costs.copy(),
        global_costs=smp._global_costs.copy(),
        node_attr_fn=smp.node_attr_fn,
        edge_attr_fn=smp.edge_attr_fn,
        missing_edge_cost_fn=smp.missing_edge_cost_fn,
        local_cost_threshold=smp.local_cost_threshold,
        global_cost_threshold=smp.global_cost_threshold,
        strict_threshold=smp.strict_threshold,
        ground_truth_provided=smp._ground_truth_provided,
        candidate_print_limit=smp._candidate_print_limit)

def satisfies_cost_threshold(smp, cost):
    # Ignore states whose cost is too high
    if smp.strict_threshold:
        if cost < smp.global_cost_threshold:
            return True
    elif cost <= smp.global_cost_threshold:
        return True
    return False

def create_new_state(smp, tmplt_idx, cand_idx, matching):
    new_matching_list = list(matching)
    bisect.insort(new_matching_list, (tmplt_idx, cand_idx))
    new_state = State()
    new_state.matching = tuple(new_matching_list)
    new_state.cost = smp.global_costs[tmplt_idx, cand_idx]
    return new_state

def impose_state_assignments_on_smp(smp, tmplt_idx, state, **kwargs):
    cand_counts = smp.candidates().sum(axis=1)
    set_fixed_costs(smp.fixed_costs, state.matching)
    # from_local_bounds(smp) # TODO: There is a chance this call makes the code slower.
    # changed_cands = smp.candidates().sum(axis=1) != cand_counts
    # TODO: Bring back reduce_world and modify the changed_cands as needed.
    # TODO: If it makes your life easier, modify the reduce_world functin to give you the index maps you need.
    # Do not reduce world as it can mess up the world indices in the matching
    changed_cands = np.zeros((smp.tmplt.n_nodes,), dtype=np.bool)
    changed_cands[tmplt_idx] = True
    iterate_to_convergence(smp, changed_cands=changed_cands, **kwargs)

def propagate_cost_threshold_changes(smp, child_smp):
    if child_smp.global_cost_threshold < smp.global_cost_threshold:
        smp.strict_threshold = child_smp.strict_threshold
        smp.global_cost_threshold = child_smp.global_cost_threshold
        iterate_to_convergence(smp, reduce_world=False, nodewise=nodewise,
                                   edgewise=edgewise)
        return True
    elif not smp.strict_threshold and child_smp.strict_threshold:
        smp.strict_threshold = True
        iterate_to_convergence(smp, reduce_world=False, nodewise=nodewise,
                                   edgewise=edgewise)
        return True
    return False

def add_new_solution(smp, solution_state, solutions, k, **kwargs):
    bisect.insort(solutions, solution_state)

    if len(solutions) == k and not smp.strict_threshold:
        kth_cost = solutions[-1].cost
        smp.global_cost_threshold = min(smp.global_cost_threshold, kth_cost)
        smp.strict_threshold = True
        iterate_to_convergence(smp, reduce_world=False, nodewise=nodewise, edgewise=edgewise)
        return True
    # TODO: Why not `k = np.inf` instead of `k = -1` for infinite matches?
    if k > 0 and len(solutions) > k:
        solutions.pop()
        kth_cost = solutions[-1].cost
        if kth_cost < smp.global_cost_threshold:
            smp.global_cost_threshold = min(smp.global_cost_threshold,
                                            kth_cost)
            iterate_to_convergence(smp, reduce_world=False, nodewise=nodewise, edgewise=edgewise)
            return True
    return False

def next_matchings(smp, state):
    candidates = smp.candidates()

    # TODO: Wrap the next few lines into a function in search_utils.py unless you reuse them.
    # Identify template node with the least number of candidates
    cand_counts = candidates.sum(axis=1)
    # TODO: Maybe update the matching with any template nodes that have only one candidate.
    # Prevent previously matched template idxs from being chosen
    matching_dict = dict_from_tuple(state.matching)
    cand_counts[list(matching_dict)] = np.max(cand_counts) + 1

    tmplt_idx = cand_counts.argmin()

    if hasattr(smp, "template_importance"):
        # The lower the number the more important the node
        # Most important nodes should be chosen first
        for new_tmplt_idx in range(smp.tmplt.n_nodes):
            if new_tmplt_idx not in matching_dict:
                curr_importance = smp.tmplt.template_importance[str(smp.tmplt.nodes[tmplt_idx])]
                new_importance = smp.tmplt.template_importance[str(smp.tmplt.nodes[new_tmplt_idx])]
                if curr_importance > new_importance and cand_counts[tmplt_idx] > cand_counts[new_tmplt_idx]:
                    tmplt_idx = new_tmplt_idx

    cand_idxs = list(np.argwhere(candidates[tmplt_idx]).flatten())

    return tmplt_idx, cand_idxs

def sort_by_cost(smp, tmplt_idx, cand_idxs):
    if len(cand_idxs) > 0:
        reorder = smp.global_costs[tmplt_idx, cand_idxs].argsort()
        cand_idxs[:] = cand_idxs[reorder]

def pop_least_cost_cand(smp, tmplt_idx, cand_idxs):
    min_idx = smp.global_costs[tmplt_idx, cand_idxs].argmin()
    return cand_idxs.pop(min_idx)

def _greedy_best_k_matching_recursive(smp, *, current_state, k,
                                      nodewise, edgewise, solutions, verbose):

    if verbose:
        # kth_cost is a bound on the cost of the k'th best match.
        kth_cost = float("inf")
        if len(solutions) == k:
            kth_cost = solutions[-1].cost  # Assume `solutions` is sorted.

        print("Current state: {} matches".format(len(current_state.matching)),
              "current_cost:", current_state.cost,
              "kth_cost:", kth_cost,  "max_cost", smp.global_cost_threshold,
              "solutions found:", len(solutions))

    # Ignore states whose cost is too high
    if not satisfies_cost_threshold(smp, current_state.cost):
        return

    # Choose the next template node to match
    tmplt_idx, cand_idxs = next_matchings(smp, current_state)
    if verbose:
        print("Choosing candidate for", tmplt_idx,
              "with {} possibilities".format(len(cand_idxs)))

    # Sort candidates for the template node by global cost bound
    # sort_by_cost(smp, tmplt_idx, cand_idxs)

    while len(cand_idxs) > 0:
        # # Pop the candidate with the lowest cost
        # cand_idx = cand_idxs[0]
        # cand_idxs = cand_idxs[1:]
        cand_idx = pop_least_cost_cand(smp, tmplt_idx, cand_idxs)

        new_state = create_new_state(smp, tmplt_idx, cand_idx, current_state.matching)

        if not satisfies_cost_threshold(smp, new_state.cost):
            break

        if len(new_state.matching) == smp.tmplt.n_nodes:
            costs_changed = add_new_solution(smp, new_state, solutions, k,
                             reduce_world=False, nodewise=nodewise, edgewise=edgewise)
        else:
            child_smp = copy_smp_shallow_graphs(smp)

            impose_state_assignments_on_smp(child_smp, tmplt_idx, new_state,
                                   reduce_world=False, nodewise=nodewise,
                                   edgewise=edgewise)

            _greedy_best_k_matching_recursive(child_smp, current_state=new_state,
                                             k=k, nodewise=nodewise,
                                             edgewise=edgewise,
                                             solutions=solutions,
                                             verbose=verbose)

            costs_changed = propagate_cost_threshold_changes(smp, child_smp)

        # if costs_changed:
            # sort_by_cost(smp, tmplt_idx, cand_idxs)

def matching_dict_from_candidates(candidates):
    matching_dict = {}
    for i in range(candidates.shape[0]):
        if np.sum(candidates[i]) == 1:
            matching_dict[i] = np.argwhere(candidates[i])[0][0]
    return matching_dict

def greedy_best_k_matching_recursive(orig_smp, k=1, nodewise=True, edgewise=True,
                                     solutions=None, verbose=False):
    if orig_smp.global_cost_threshold == float("inf"):
        raise Exception("Invalid global cost threshold.")
    # Initialize matching with known matches
    if solutions is None:
        solutions = []

    # TODO: We don't actually need the state option for the recursive implementation.
    # We could derive the matching from the smp by looking at which rows of the
    # candidates matrix have only one nonzero and extractive the corresponding global costs.
    # TODO: To avoid recomputing the matching all the time in the idea above,
    # the smp can carry around some notion of `matching` as a dict from template
    # nodes that only have one candidate to that corresponding candidate.

    smp = copy_smp_shallow_graphs(orig_smp)
    current_state = State()  # Consider initializing this at the end of the block with arguments of `matching` and `cost`
    candidates = smp.candidates()

    matching_dict = matching_dict_from_candidates(candidates)
    current_state.matching = tuple_from_dict(matching_dict)

    current_state.cost = max((smp.global_costs[x] for x in current_state.matching),
                             default=smp.global_costs.min())

    # Handle the case where we start in a solved state
    if len(current_state.matching) == smp.tmplt.n_nodes and len(solutions) == 0:
        solutions.append(current_state)
        return solutions

    changed_cands = np.zeros((smp.tmplt.n_nodes,), dtype=np.bool)
    for tmplt_idx, cand_idx in current_state.matching:
        changed_cands[tmplt_idx] = True
    smp.enforce_matching(current_state.matching)

    iterate_to_convergence(smp, changed_cands=changed_cands, reduce_world=False,
                           nodewise=nodewise,
                           edgewise=edgewise)

    _greedy_best_k_matching_recursive(smp, current_state=current_state,
                                      k=k, nodewise=nodewise, edgewise=edgewise,
                                      solutions=solutions, verbose=verbose)
    return solutions
