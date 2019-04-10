import numpy as np
import networkx as nx
from .all_filters import all_filters
from ..utils.misc import one_hot

def centrality_ordered_node_idxs(tmplt):
    """
    Use some arbitrary heuristics to sort the idxs of the template nodes.
    """
    
    # construct a dict mapping node_idx to the largest k for which it appears
    # in the k-core of the template
    node_idx_to_max_k = {}

    cand_counts = tmplt.is_cand.sum(axis=1)
    degrees = tmplt.sym_composite_adj.sum(axis=1)
    nbr_counts = tmplt.is_nbr.sum(axis=1)

    # The less candidates the better, and the more important the node is
    metric_cand_count = lambda node_idx: cand_counts[node_idx]

    # The higher the degree, the more important the node.
    metric_degree = lambda node_idx: -degrees[node_idx]

    # The higher the max-k-core of the node, the more important it is
    metric_nbr_count = lambda node_idx: -nbr_counts[node_idx]
    
    # TODO: optimize over metric orders
    # Put the metrics in some arbitrary order
    metrics = [metric_cand_count, metric_nbr_count, metric_degree]
    metric_tuple = lambda idx: tuple(metric(idx) for metric in metrics)

    return sorted(range(tmplt.n_nodes), key=metric_tuple)

def elimination_filter(tmplt, world,
                       changed_nodes=None,
                       verbose=False,
                       **kwargs):
    """
    If choosing a candidate for a template node and running the the filters
    results in all of the candidates disappearing, that candidate is
    eliminated
    """
    nbr_counts = tmplt.is_nbr.sum(axis=1).flat

    n_skipped = 0
    for node_idx in centrality_ordered_node_idxs(tmplt):
        if (changed_nodes is not None) and not changed_nodes[node_idx]:
            continue
        
        n_candidates = np.sum(tmplt.is_cand[node_idx])
        # If the node only has one candidate, there is no need to check it
        # If the node only has one neighbor, there is no point in filtering on
        # it since it will be taken care of by filtering on its one neighbor
        if n_candidates == 1 or nbr_counts[node_idx] == 1:
            # print("skipping", tmplt.nodes[node_idx])
            n_skipped += 1
            continue
        
        if verbose:
            print("trying", tmplt.nodes[node_idx], "which has",
                  n_candidates, "candidates")
                  
        changed_nodes = np.zeros(tmplt.nodes.shape)

        cand_idxs = np.argwhere(tmplt.is_cand[node_idx]).flat
        for i, cand_idx in enumerate(cand_idxs):
            # Don't modify the original template unless you mean to
            tmplt_copy = tmplt.copy()
            tmplt_copy.is_cand[node_idx, :] = one_hot(cand_idx, tmplt.n_cands)
                
            if verbose and i % 10 == 0:
                print("cand {} of {}".format(i, len(cand_idxs)))
                
            all_filters(tmplt_copy, world, elimination=False, verbose=False,
                        initial_changed_nodes=one_hot(node_idx, tmplt.n_nodes))

            # TODO: add something to the data structure so we can check this
            # without have to do the summation every time
            if np.sum(tmplt_copy.is_cand) == 0:
                tmplt.is_cand[node_idx, cand_idx] = False
                changed_nodes[node_idx] = True

        if np.any(changed_nodes):
            all_filters(tmplt, world, elimination=False,
                        initial_changed_nodes=changed_nodes, verbose=False)
            if verbose:
                tmplt.summarize()
    if verbose:
        print("Elimination filter finished, skipped {} nodes".format(n_skipped))
