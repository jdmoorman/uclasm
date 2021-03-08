"""Provide a function for bounding global assignment costs from local costs."""
from laptools import clap
import numpy as np


def from_local_bounds(smp):
    """Bound global costs by smallest linear sum assignment of local costs.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute nodewise cost bounds.
    """
    if smp.match_fixed_costs:
        costs = smp.local_costs / 2 + smp.fixed_costs
        global_cost_bounds = clap.costs(costs)
        smp.global_costs[:] = global_cost_bounds
    else:
        tmplt_idx_mask = np.ones(smp.tmplt.n_nodes, dtype=np.bool)
        world_idx_mask = np.ones(smp.world.n_nodes, dtype=np.bool)
        for tmplt_idx, world_idx in smp.matching:
            tmplt_idx_mask[tmplt_idx] = False
            world_idx_mask[world_idx] = False
            if (tmplt_idx, world_idx) in smp.prevented_matches:
                print("Matching includes a prevented match!")
        # Prevent matches by setting their local cost to infinity
        for tmplt_idx, world_idx in smp.prevented_matches:
            smp.local_costs[tmplt_idx, world_idx] = float("inf")

        partial_match_cost = np.sum([smp.local_costs[match]/2 + smp.fixed_costs[match] for match in smp.matching])
        mask = np.ix_(tmplt_idx_mask, world_idx_mask)
        total_match_cost = partial_match_cost
        if np.any(tmplt_idx_mask):
            costs = smp.local_costs[mask] / 2 + smp.fixed_costs[mask]
            global_cost_bounds = clap.costs(costs)
            smp.global_costs[mask] = global_cost_bounds + partial_match_cost
            total_match_cost += np.min(global_cost_bounds)
        non_matching_mask = smp.get_non_matching_mask()
        smp.global_costs[non_matching_mask] = float("inf")
        for tmplt_idx, world_idx in smp.matching:
            smp.global_costs[tmplt_idx, world_idx] = total_match_cost

    # TODO: should the global costs for local costs introduced by neighborhood
    # constraint be computed in the same way?
