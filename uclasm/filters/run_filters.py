import copy
import time
import numpy as np
from . import label_filter
from . import permutation_filter
from ..utils import summarize

# TODO: logging

def run_filters(tmplt, world, *,
                candidates=None,
                filters=None,
                verbose=False,
                max_iter=-1,
                init_changed_cands=None):
    """
    Repeatedly run the desired filters until the candidates converge
    """

    has_gt = len(set(tmplt.nodes) - set(world.nodes)) == 0

    # Boolean matrix with i,j entry denoting whether world node j is a candidate
    # for template node i
    if candidates is None:
        candidates = np.ones((tmplt.n_nodes, world.n_nodes), dtype=np.bool)
    label_filter(tmplt, world, candidates)

    if filters is None:
        from . import all_filters
        filters = all_filters

    # Construct a list of the filters that have been run for reference when
    # evaluating performance. This is appended to repeatedly in the loop below
    filters_so_far = []

    # index in `filters` of the filter we should run next
    filter_idx = 0

    # Each element of this list is a copy `cand_counts` from before the last
    # time the corresponding filter was run.
    old_cand_counts_list = [None for filter in filters]
    cand_counts = candidates.sum(axis=1)
    init_cand_counts = candidates.sum(axis=1)

    if init_changed_cands is None:
        init_changed_cands = np.ones(tmplt.nodes.shape, dtype=np.bool)

    changed_cands = init_changed_cands

    while filter_idx != len(filters) and len(filters_so_far) != max_iter:
        filter = filters[filter_idx]

        # Get cand counts from before last run of this filter
        old_cand_counts = old_cand_counts_list[filter_idx]

        # Update the cand counts for the current filter for next time it runs
        old_cand_counts_list[filter_idx] = cand_counts

        # Find the nodes whose candidates have changed since last time this
        # filter was run
        if old_cand_counts is not None:
            changed_cands = cand_counts < old_cand_counts
        else:
            changed_cands = init_changed_cands | (cand_counts < init_cand_counts)

        # If any template nodes have candidates that have changed since the
        # last time this filter was run, go ahead and run the filter.
        if np.any(changed_cands):
            # TODO: create an object we can use like `with Timer()`
            start_time = time.time()

            # TODO: we could get rid of the `verbose` flag using a singleton
            # logger that stores a global `verbose` property
            if verbose:
                print("running", filter.__name__)

            # Run whatever filter and the permutation filter
            tmplt, world, candidates = filter(
                tmplt, world, candidates, changed_cands=changed_cands,
                verbose=verbose)
            filters_so_far.append(filter.__name__.replace("_filter", ""))
            tmplt, world, candidates = permutation_filter(
                tmplt, world, candidates)

            # TODO: make logging less cumbersome
            if verbose:
                end_time = time.time()
                summarize(tmplt, world, candidates, alert_missing=has_gt)
                print("after", filter.__name__,
                      "on iteration", len(filters_so_far),
                      "took", end_time - start_time, "seconds")
                print("filters so far: {}".format(" ".join(filters_so_far)))

        cand_counts_after_filter = candidates.sum(axis=1)

        # If any candidates have changed, start over from the first filter.
        # Otherwise, move on to the next filter in the list on the next pass.
        if np.any(cand_counts_after_filter < cand_counts):
            filter_idx = 0
        else:
            filter_idx += 1

        cand_counts = cand_counts_after_filter

        # If some template node has no candidates, there are no isomorphisms
        if np.any(cand_counts == 0):
            candidates[:,:] = False
            break

        # Which world nodes are candidates for at least one template node?
        is_cand_any = candidates.any(axis=0)

        # If not all world nodes are candidates for at least one template node
        if ~is_cand_any.all():
            # Get rid of unnecessary world nodes
            world = world.subgraph(is_cand_any)
            candidates = candidates[:, is_cand_any]

    if verbose:
        print("filters are done.")

    return tmplt, world, candidates
