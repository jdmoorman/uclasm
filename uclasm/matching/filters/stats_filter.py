from ..cost_bounds.nodewise_cost_bound import feature_disagreements
import numpy as np

def stats_filter(smp):
    """Filtering based on statistics.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute nodewise cost bounds.
    """
    disagreements = feature_disagreements(smp.tmplt.in_out_degrees,
                                          smp.world.in_out_degrees)

    is_cand = np.logical_and(disagreements <= smp.global_cost_threshold,
                             disagreements <= smp.local_cost_threshold)

    smp.update_costs(np.Inf, ~is_cand)
