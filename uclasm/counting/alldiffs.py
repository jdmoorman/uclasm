from ..utils.misc import invert, values_map_to_same_key
import numpy as np
from functools import reduce


# counts the number of ways to assign a particular node, recursing to
# incorporate ways to assign the next node, and so on
def recursive_alldiff_counter(node_to_nodes_list, nodes_to_cand_counts):
    # no nodes left to assign
    if len(node_to_nodes_list) == 0:
        return 1

    count = 0

    # give me an arbitrary unspecified nodeiable
    node, nodes_list = node_to_nodes_list.popitem()

    # for each way of assigning the given nodeiable
    for nodes in nodes_list:
        # how many ways to assign the nodeiable in this way are there?
        n_cands = nodes_to_cand_counts[nodes]
        if n_cands == 0:
            continue

        nodes_to_cand_counts[nodes] -= 1

        # number of ways to assign current node times number of ways to
        # assign the rest
        n_ways_to_assign_rest = recursive_alldiff_counter(
            node_to_nodes_list, nodes_to_cand_counts)

        count += n_cands * n_ways_to_assign_rest

        # put the count back so we don't mess up the recursion
        nodes_to_cand_counts[nodes] += 1

    # put the list back so we don't mess up the recursion
    node_to_nodes_list[node] = nodes_list

    return count

def count_alldiffs(node_to_cands):
    """
    node_to_cands: dict(item, list)

    count the number of ways to assign nodes to cands without using any cand for
    more than one node. ie. count solns to alldiff problem, where the nodeiables
    are the keys of node_to_cands, and the domains are the values.
    """

    # TODO: can this function be vectorized?
    # TODO: does scipy have a solver for this already?

    # Check if any node has no cands
    if any(len(cands)==0 for cands in node_to_cands.values()):
        return 0

    # TODO: throwing out nodes with only one cand may not be necessary
    # if a node has only one possible cand, throw it out.
    node_to_cands = {node: cands for node, cands in node_to_cands.items()
                   if len(cands) > 1}

    unspec_nodes = list(node_to_cands.keys())

    # which nodes is each cand a cand for?
    cand_to_nodes = invert(node_to_cands)

    # gather sets of cands which have the same set of possible nodes.
    nodes_to_cands = values_map_to_same_key(cand_to_nodes)
    nodes_to_cand_counts = {nodes: len(cands)
                          for nodes, cands in nodes_to_cands.items()}

    # each node can belong to multiple sets of nodes which key nodes_to_cand_counts
    # so here we find out which sets of nodes each node belongs to
    node_to_nodes_list = {
        node: [nodes for nodes in nodes_to_cand_counts.keys() if node in nodes]
        for node in node_to_cands}

    return recursive_alldiff_counter(node_to_nodes_list, nodes_to_cand_counts)
