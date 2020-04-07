import numpy as np

from .matching import MatchingProblem
from .matching import local_cost_bound, global_cost_bound

# Dummy variable for old code that needs it
all_filters = None

def run_filters(tmplt, world, *, candidates=None, filters=None, verbose=False):
    """Interface with old format for calling filters. Assumes exact matching
    with cost bound of zero.

    Parameters
    ----------
    tmplt : Graph
        A Graph object
    world : Graph
        A Graph object
    candidates : ndarray(bool)
        Candidates to initialize with
    filters : list
        Previously, the list of filters to run. Currently ignored.
    verbose : bool
        Flag for verbose output
    """
    # Configure the fixed costs to enforce the given candidates
    if candidates is None:
        smp = MatchingProblem(tmplt, world)
    else:
        fixed_costs = np.zeros(candidates.shape)
        fixed_costs[~candidates] = float("inf")
        smp = MatchingProblem(tmplt, world, fixed_costs=fixed_costs)
    smp.global_costs = smp.local_costs/2 + smp.fixed_costs
    while True:
        old_candidates = smp.candidates().copy()
        print(smp)
        print("Running nodewise cost bound")
        local_cost_bound.nodewise(smp)
        smp.global_costs = smp.local_costs/2 + smp.fixed_costs
        print(smp)
        print("Running global cost bound")
        global_cost_bound.from_local_bounds(smp)
        print(smp)
        print("Running edgewise cost bound")
        local_cost_bound.edgewise(smp)
        smp.global_costs = smp.local_costs/2 + smp.fixed_costs
        print(smp)
        print("Running global cost bound")
        global_cost_bound.from_local_bounds(smp)
        print(smp)
        if np.all(smp.candidates() == old_candidates):
            break
        print(smp)
    return tmplt, world, smp.candidates()
