import numpy as np

from .matching import MatchingProblem
from .matching import local_cost_bound, global_cost_bound
from .matching import search

# Dummy variable for old code that needs it
all_filters = None

def run_filters_old(tmplt, world, *, candidates=None, filters=None, verbose=False):
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
    search.search_utils.iterate_to_convergence(smp)
    return smp.tmplt, smp.world, smp.candidates()

def count_isomorphisms(tmplt, world, candidates=None, verbose=False):
    """Temporary interface for old code. Counts the number of isomorphisms.
    Likely incredibly inefficient.

    Parameters
    ----------
    tmplt : Graph
        A Graph object
    world : Graph
        A Graph object
    candidates : ndarray(bool)
        Candidates to initialize with
    verbose : bool
        Flag for verbose output
    """
    if candidates is None:
        tmplt, world, candidates = run_filters_old(tmplt, world, verbose=verbose)
    fixed_costs = np.zeros(candidates.shape)
    fixed_costs[~candidates] = float("inf")
    smp = MatchingProblem(tmplt, world, fixed_costs=fixed_costs)
    local_cost_bound.nodewise(smp)
    local_cost_bound.edgewise(smp)
    global_cost_bound.from_local_bounds(smp)

    iso_list = search.greedy_best_k_matching(smp, k=-1, verbose=verbose)

    return len(iso_list)
