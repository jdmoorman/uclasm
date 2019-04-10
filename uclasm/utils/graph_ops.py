


# TODO: get rid of reliance on networkx through nxgraph

def get_unspec_cover(tmplt):
    """
    get a reasonably small set of template nodes which, if removed, would cause
    all of the remaining template nodes with multiple candidates to become
    disconnected
    """

    unspec_node_idxs = [list(tmplt.nodes).index(node)
                    for node, cands in tmplt.candidate_sets.items()
                    if len(cands) > 1]
    unspec_subgraph = tmplt._nxgraph.subgraph(unspec_node_idxs)
    nxgraph = unspec_subgraph.to_undirected()
    node_cover = []
    while nxgraph.number_of_edges() > 0:
        degrees = {n: d for n, d in nxgraph.degree()}
        max_nbr_node = max(nxgraph.nodes,
                           key=lambda n:
                                (len(list(nxgraph.neighbors(n))),
                                 -len(tmplt.candidate_sets[tmplt.nodes[n]])))
        node_cover.append(max_nbr_node)
        nxgraph.remove_node(max_nbr_node)
    return tmplt.nodes[node_cover]
