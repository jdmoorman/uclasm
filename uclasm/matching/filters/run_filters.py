import time
import numpy as np
from . import stats_filter
from . import topology_filter

# Note: this run_filters is for testing purposes

def run_filters(smp, verbose=True):
    """
    Repeatedly run the desired filters until the candidates converge
    """

    num_iter = 0

    while smp._have_candidates_changed() or num_iter == 0:
        stats_filter(smp)
        topology_filter(smp)
        smp.reduce_world()
        num_iter += 1

    if verbose:
        print(smp)
        print("filters are done after {} iterations.".format(num_iter))
