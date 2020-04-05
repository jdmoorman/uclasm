from ..local_cost_bound.edgewise import edgewise_local_cost
import numpy as np

def topology_filter(smp):
    """Filtering based on topology.

    TODO: Add the option of filtering just for one node

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute nodewise cost bounds.
    """
    disagreements = edgewise_local_cost(smp)

    is_cand = np.logical_and(disagreements <= smp.global_cost_threshold,
                             disagreements <= smp.local_cost_threshold)

    # TODO: check whether this works
    smp.local_costs[~is_cand] = np.Inf
