import numpy as np
import time

# TODO: can we use changed_cands?

def compute_features(graph, channels=None):
    """
    is_cand_any is a bool vector of nodes we care about
    """

    features_list = []

    # default to computing features for every channel in the default order
    if channels is None:
        channels = graph.channels

    # for each channel, compute some featurers
    for channel in channels:
        adj = graph.ch_to_adj[channel]

        # in degree
        features_list.append(adj.sum(axis=0).A)

        # out degree
        features_list.append(adj.sum(axis=1).T.A)

        # max in edge multiplicity
        features_list.append(adj.max(axis=0).A)

        # max out edge multiplicity
        features_list.append(adj.max(axis=1).todense().T.A)

        # neighbors coming in
        features_list.append((adj > 0).sum(axis=0).A)

        # neighbors going out
        features_list.append((adj > 0).sum(axis=1).T.A)

        # self edges
        features_list.append(np.reshape(adj.diagonal(), (1, -1)))

        # reciprocated edges
        features_list.append(adj.multiply(adj.T).sum(axis=0).A)

    return np.concatenate(features_list, axis=0)


def compute_features_dense(graph, channels=None):
    """
    This applies stats filter to dense matrices
    """

    features_list = []

    # default to computing features for every channel in the default order
    if channels is None:
        channels = graph.channels

    # for each channel, compute some featurers
    for channel in channels:
        adj = graph.ch_to_adj[channel]

        # in degree
        features_list.append(adj.sum(axis=0))

        # out degree
        features_list.append(adj.sum(axis=1).T)

        # max in edge multiplicity
        features_list.append(adj.max(axis=0))

        # max out edge multiplicity
        features_list.append(adj.max(axis=1).T)

        # neighbors coming in
        features_list.append((adj > 0).sum(axis=0))

        # neighbors going out
        features_list.append((adj > 0).sum(axis=1).T)

        # self edges
        features_list.append(adj.diagonal())

        # reciprocated edges
        features_list.append((adj * adj.T).sum(axis=0))

    return np.vstack(features_list)


class _cache():
    tmplt = None
    tmplt_feats = None
    world = None
    world_feats = None

def stats_filter(tmplt, world, candidates, *, verbose=False, noise_level=0, dense=False, **kwargs):
    global _cache

    if tmplt == _cache.tmplt:
        tmplt_feats = _cache.tmplt_feats
    else:
        if dense:
            tmplt_feats = compute_features_dense(tmplt)
        else:
            tmplt_feats = compute_features(tmplt)
        _cache.tmplt = tmplt
        _cache.tmplt_feats = tmplt_feats

    if world == _cache.world:
        world_feats = _cache.world_feats
    else:
        if dense:
            world_feats = compute_features_dense(world, channels=tmplt.channels)
        else:
            world_feats = compute_features(world, channels=tmplt.channels)
        _cache.world = world
        _cache.world_feats = world_feats

    for tmplt_node_idx, tmplt_node in enumerate(tmplt.nodes):
        tmplt_node_feats = tmplt_feats[:, [tmplt_node_idx]]
        new_is_cand = np.all(world_feats >= tmplt_node_feats - noise_level, axis=0)
        candidates[tmplt_node_idx] &= new_is_cand

    return tmplt, world, candidates
