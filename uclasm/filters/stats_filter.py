import numpy as np
import time

# TODO: can we use changed_nodes?
# TODO: handle world not being reduced to candidates

def compute_features(graph, channels=None, is_cand_any=None):
    """
    is_cand_any is a bool vector of nodes we care about
    """
    features = []

    # default to computing features for every channel in the default order
    if channels is None:
        channels = graph.channels

    # for each channel, compute some featurers
    for channel in channels:
        adj = graph.ch_to_adj[channel]
        
        if is_cand_any is not None:
            adj = adj[is_cand_any, :][:, is_cand_any]

        # in degree
        features.append(adj.sum(axis=0))

        # out degree
        features.append(adj.sum(axis=1).T)

        # max in edge multiplicity
        features.append(adj.max(axis=0).todense())

        # max out edge multiplicity
        features.append(adj.max(axis=1).todense().T)

        # neighbors coming in
        features.append((adj > 0).sum(axis=0))

        # neighbors going out
        features.append((adj > 0).sum(axis=1).T)

        # self edges
        features.append(np.reshape(adj.diagonal(), (1, -1)))

        # reciprocated edges
        features.append(adj.multiply(adj.T).sum(axis=0))

    return np.stack(features)

def stats_filter(tmplt, world, verbose=False, **kwargs):
    # Boolean array indicating if a given world node is a candidate for any
    # template node. If a world node is not a candidate for any template nodes,
    # we shouldn't bother calculating its features.
    is_cand_any = np.any(tmplt.is_cand, axis=0)
    
    # No candidates for any template node
    if np.sum(is_cand_any) == 0: 
        return

    tmplt_feats = compute_features(tmplt)
    world_feats = compute_features(world, channels=tmplt.channels,
                                   is_cand_any=is_cand_any)

    for tmplt_node_idx, tmplt_node in enumerate(tmplt.nodes):
        tmplt_node_feats = tmplt_feats[:, tmplt_node_idx]
        new_is_cand = np.all(world_feats >= tmplt_node_feats, axis=0)
        
        # TODO: transpose the features so we don't need this .flat
        tmplt.is_cand[tmplt_node_idx,is_cand_any] &= new_is_cand.flat
