import numpy as np
import time

# TODO: can we use changed_cands?
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
        features.append(adj.sum(axis=0).A)

        # out degree
        features.append(adj.sum(axis=1).T.A)

        # max in edge multiplicity
        features.append(adj.max(axis=0).A)

        # max out edge multiplicity
        features.append(adj.max(axis=1).todense().T.A)

        # neighbors coming in
        features.append((adj > 0).sum(axis=0).A)

        # neighbors going out
        features.append((adj > 0).sum(axis=1).T.A)

        # self edges
        features.append(np.reshape(adj.diagonal(), (1, -1)))

        # reciprocated edges
        features.append(adj.multiply(adj.T).sum(axis=0).A)

    return np.concatenate(features, axis=0)

def stats_filter(tmplt, world, candidates, *,
                 verbose=False, **kwargs):
    # Boolean array indicating if a given world node is a candidate for any
    # template node. If a world node is not a candidate for any template nodes,
    # we shouldn't bother calculating its features.
    is_cand_any = np.any(candidates, axis=0)

    # No candidates for any template node
    if np.sum(is_cand_any) == 0:
        return

    tmplt_feats = compute_features(tmplt)
    world_feats = compute_features(world, channels=tmplt.channels,
                                   is_cand_any=is_cand_any)

    for tmplt_node_idx, tmplt_node in enumerate(tmplt.nodes):
        tmplt_node_feats = tmplt_feats[:, [tmplt_node_idx]]
        new_is_cand = np.all(world_feats >= tmplt_node_feats, axis=0)

        candidates[tmplt_node_idx,is_cand_any] &= new_is_cand
