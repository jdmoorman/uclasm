import time
import numpy as np
from . import stats_filter
from . import topology_filter
from ..global_cost_bound import *

# Note: this run_filters is for testing purposes

def run_filters(smp, verbose=True):
    """
    Repeatedly run the desired filters until the candidates converge
    """
    num_iter = 0
    # Note: most efficient if we only call from_local_bounds in reduce_world
    while smp.have_candidates_changed():
        stats_filter(smp)
        # from_local_bounds(smp)
        topology_filter(smp)
        # from_local_bounds(smp)
        smp.reduce_world()
        print("There are {} nodes left in the world.".format(smp.world.n_nodes))
        num_iter += 1

    if verbose:
        print(smp)
        print("filters are done after {} iterations.".format(num_iter))
