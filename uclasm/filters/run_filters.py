import copy
import time
import numpy as np
from .permutation_filter import permutation_filter

def run_filters(tmplt, world, filters,
                verbose=False,
                max_iter=-1,
                initial_changed_nodes=None):
    """
    Repeatedly run the desired filters until the candidates converge
    """

    # Construct a list of the filters that have been run for reference when
    # evaluating performance. This is appended to repeatedly in the loop below
    filters_so_far = []

    # index in `filters` of the filter we should run next
    filter_idx = 0

    # Each element of this list is a copy tmplt.is_cand from before the last
    # time the corresponding filter was run.
    old_cand_counts_list = [None for filter in filters]
    cand_counts = tmplt.get_cand_counts()

    if initial_changed_nodes is None:
        initial_changed_nodes = np.ones(tmplt.nodes.shape, dtype=np.bool)

    while filter_idx != len(filters) and len(filters_so_far) != max_iter:
        filter = filters[filter_idx]

        # TODO: rename old_cand_counts to something more descriptive
        # Update the cand counts for the current filter for next time it runs
        old_cand_counts = old_cand_counts_list[filter_idx]
        old_cand_counts_list[filter_idx] = cand_counts

        # Find the nodes whose candidates have changed since last time this
        # filter was run
        if old_cand_counts is not None:
            changed_nodes = cand_counts < old_cand_counts
        else:
            changed_nodes = initial_changed_nodes

        # If any template nodes have candidates that have changed since the
        # last time this filter was run, go ahead and run the filter.
        if np.any(changed_nodes):
            # TODO: create an object we can use like `with Timer()`
            start_time = time.time()

            # TODO: we could get rid of the `verbose` flag using a singleton
            # logger that stores a global `verbose` property
            if verbose:
                print("running", filter.__name__)

            # Run whatever filter and the permutation filter
            filter(tmplt, world, changed_nodes=changed_nodes, verbose=verbose)
            filters_so_far.append(filter.__name__.replace("_filter", ""))
            permutation_filter(tmplt, world)

            # TODO: make logging less cumbersome
            if verbose:
                end_time = time.time()
                tmplt.summarize()
                print("after", filter.__name__,
                      "on iteration", len(filters_so_far),
                      "took", end_time - start_time, "seconds")
                print("filters so far: {}".format(" ".join(filters_so_far)))

        cand_counts_after_filter = tmplt.get_cand_counts()

        # If any candidates have changed, start over from the first filter.
        # Otherwise, move on to the next filter in the list on the next pass.
        if np.any(cand_counts_after_filter < cand_counts):
            filter_idx = 0
        else:
            filter_idx += 1

        cand_counts = cand_counts_after_filter

        # If some template node has no candidates, there are no isomorphisms
        if np.any(cand_counts == 0):
            tmplt.is_cand[:,:] = False
            break

    if verbose:
        print("filters are done.")
