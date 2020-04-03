from ..cost_bounds.edgewise_cost_bound import edge_disagreements
import numpy as np

def topology_filter(smp):
    """Filtering based on topology.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute nodewise cost bounds.
    """
    disagreements = edge_disagreements(smp)

    is_cand = np.logical_and(disagreements <= smp.global_cost_threshold,
                             disagreements <= smp.local_cost_threshold)

    smp.update_costs(np.Inf, ~is_cand)
