import copy
import time
import numpy as np

def all_filters(tmplt, world, 
                stats=True,
                topology=True,
                neighborhood=False,
                elimination=False, 
                permutation=True,
                verbose=False,
                reduce_world=True,
                max_iter=-1,
                initial_changed_nodes=None):
    """
    
    """
    # Construct a list of the desired filters from least to most expensive
    filters = []
    if stats:
        from .stats_filter import stats_filter
        filters.append(stats_filter)
    if topology:
        from .topology_filter import topology_filter
        filters.append(topology_filter)
    # if neighborhood:
    #     from .neighborhood_filter import neighborhood_filter
    #     filters.append(neighborhood_filter)
    if elimination:
        from .elimination_filter import elimination_filter
        filters.append(elimination_filter)
        
    # Permutation filter runs after every other filter, so no need to append
    # it to the list of filters
    if permutation:
        from .permutation_filter import permutation_filter

    # Construct a list of the filters that have been run for reference when
    # evaluating performance. This is appended to repeatedly in the loop below
    filters_so_far = []
    
    # index in `filters` of the filter we should run next
    filter_idx = 0

    # Each element of this list is a copy tmplt.is_cand from before the last
    # time the corresponding filter was run.
    old_cand_counts_list = [None for filter in filters]
    
    if initial_changed_nodes is None:
        initial_changed_nodes = np.ones(tmplt.nodes.shape, dtype=np.bool)
    
    while filter_idx != len(filters) and len(filters_so_far) != max_iter:
        filter = filters[filter_idx]

        cand_counts = tmplt.get_cand_counts()

        # If some template node has no candidates, there are no isomorphisms
        if np.any(cand_counts == 0):
            tmplt.is_cand[:,:] = False
            break
        
        # Update the cand counts for the current filter for next time it runs
        old_cand_counts = old_cand_counts_list[filter_idx]
        
        # Find the nodes whose candidates have changed since last time this
        # filter was run
        if old_cand_counts is not None:
            changed_nodes = old_cand_counts > cand_counts
        else:
            changed_nodes = initial_changed_nodes
        
        # TODO: create an object we can use like `with Timer()`
        start_time = time.time()
        
        # TODO: we could get rid of the `verbose` flag using a singleton
        # logger that stores a global `verbose` property
        if verbose:
            print("running", filter.__name__)
            
        # Run whatever filter and the permutation filter
        filter(tmplt, world, changed_nodes=changed_nodes, verbose=verbose)
        filters_so_far.append(filter.__name__.replace("_filter", ""))
        if permutation:
            permutation_filter(tmplt, world)
            # Omit permutation filter from the list of filters run so far
            
        old_cand_counts_list[filter_idx] = tmplt.get_cand_counts()

        # If any candidates have changed, start over from the first filter.
        # Otherwise, move on to the next filter in the list on the next pass.
        if np.any(changed_nodes):
            filter_idx = 0
        else:
            filter_idx += 1
            
        # TODO: make logging less cumbersome
        if verbose:
            end_time = time.time()
            tmplt.summarize()
            print("after", filter.__name__,
                  "on iteration", len(filters_so_far),
                  "took", end_time - start_time, "seconds")
            print("filters so far: {}".format(" ".join(filters_so_far)))

    if verbose:
        print("filters are done.")
