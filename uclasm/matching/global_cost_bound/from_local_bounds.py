"""Provide a function for bounding global assignment costs from local costs."""
from ..lsap import constrained_lsap_costs


def from_local_bounds(smp):
    """Bound global costs by smallest linear sum assignment of local costs.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute nodewise cost bounds.
    """
    costs = smp.local_costs / 2 + smp.fixed_costs
    global_cost_bounds = constrained_lsap_costs(costs)

    # To check: Since local_costs and fixed_costs are monotone,
    # global_cost_bounds must be automatically monotone as well.
    smp.set_costs(global_costs=global_cost_bounds)

    # TODO: should the global costs for local costs introduced by neighborhood
    # constraint be computed in the same way?
