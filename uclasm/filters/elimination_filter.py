import numpy as np
import networkx as nx
from . import run_filters, cheap_filters
from ..utils.misc import one_hot

def centrality_ordered_node_idxs(tmplt, world, candidates):
    """
    Use some arbitrary heuristics to sort the idxs of the template nodes.
    """

    cand_counts = candidates.sum(axis=1)
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

def elimination_filter(tmplt, world, candidates, *,
                       changed_cands=None,
                       verbose=False,
                       **kwargs):
    """
    If choosing a candidate for a template node and running the the filters
    results in all of the candidates disappearing, that candidate is
    eliminated
    """
    nbr_counts = tmplt.is_nbr.sum(axis=1).flat

    n_skipped = 0
    for node_idx in centrality_ordered_node_idxs(tmplt, world, candidates):
        n_candidates = np.sum(candidates[node_idx])
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

        init_changed_cands = np.zeros(tmplt.nodes.shape, dtype=np.bool)

        elim_count = 0
        cand_idxs = np.argwhere(candidates[node_idx]).flat
        for i, cand_idx in enumerate(cand_idxs):
            # Don't modify the original template unless you mean to
            candidates_copy = candidates.copy()
            candidates_copy[:, cand_idx] = False
            candidates_copy[node_idx, :] = one_hot(cand_idx, world.n_nodes)

            if verbose and i % 10 == 0:
                print("cand {} of {}".format(i, len(cand_idxs)))

            _, _, result_candidates = run_filters(
                tmplt, world, candidates=candidates_copy, filters=cheap_filters,
                init_changed_cands=one_hot(node_idx, tmplt.n_nodes),
                verbose=False)

            # TODO: add something to the data structure so we can check this
            # without have to do the summation every time
            if ~np.all(result_candidates.any(axis=1)):
                elim_count += 1
                candidates[node_idx, cand_idx] = False
                init_changed_cands[node_idx] = True

        if np.any(init_changed_cands):
            tmplt, world, candidates = run_filters(
                tmplt, world, candidates=candidates, filters=cheap_filters,
                init_changed_cands=init_changed_cands, verbose=False)
        if verbose:
            print("Eliminating", elim_count, "of", n_candidates, ";world now has", world.n_nodes, "nodes")

    if verbose:
        print("Elimination filter finished, skipped {} nodes".format(n_skipped))

    return tmplt, world, candidates
