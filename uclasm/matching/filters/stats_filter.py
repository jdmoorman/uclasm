from ..matching_utils import feature_disagreements
import numpy as np

def stats_filter(exact_smp, changed_cands=None, verbose=False):
    """Filtering based on statistics.

    Parameters
    ----------
    exact_smp : ExactMatchingProblem
        A subgraph matching problem on which to use nodewise stats filter.
    """
    disagreements = feature_disagreements(exact_smp.tmplt.in_out_degrees,
                                          exact_smp.world.in_out_degrees)

    is_cand = disagreements <= 0

    candidates = exact_smp.candidates
    candidates[~is_cand] = False
