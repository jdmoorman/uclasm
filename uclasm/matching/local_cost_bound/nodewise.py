"""Provide a function for bounding node assignment costs with nodewise info."""

import numpy as np
import numba


@numba.njit(parallel=True)
def feature_disagreements(tmplt_features, world_features):
    """Compute the amount by which the template's features exceed the world's.

    The feature disagreement is computed separately in each channel and summed.

    Parameters
    ----------
    tmplt_features : 2darray
        [n_tmplt_nodes, n_features] array of features for each template node.
    world_features : 2darray
        [n_world_nodes, n_features] array of features for each world node.

    Returns
    -------
    2darray
        [n_tmplt_nodes, n_world_nodes] array of amounts by which each template
        node's features exceeded each world node's features summed across all
        of the features.

    Notes
    -----
    Do not be tempted to use a tensorized implementation of this function.
    It will cause the memory usage of this function to explode leading to
    slower performance.
    """
    n_tmplt_nodes = tmplt_features.shape[0]
    n_world_nodes = world_features.shape[0]

    # Preallocate memory
    disagreements = np.empty((n_tmplt_nodes, n_world_nodes))

    # For each template and world node, compute the disagreement between their
    # features. Currently, the amount by which the template features exceed
    # the world features.
    for tidx in numba.prange(n_tmplt_nodes):
        tnode_features = tmplt_features[tidx, :]

        for widx in numba.prange(n_world_nodes):
            wnode_features = world_features[widx, :]

            # TODO: Insert generic loss function between features here
            disagreement = np.maximum(tnode_features - wnode_features, 0).sum()
            disagreements[tidx, widx] = disagreement

    return disagreements


def nodewise(smp):
    """Bound local assignment costs by comparing in and out degrees.

    TODO: Cite paper from REU.
    TODO: Take candidacy into account when computing features.
    TODO: Bring back reciprocated edges, carefully?

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute nodewise cost bounds.
    """
    smp.local_costs = feature_disagreements(
        smp.tmplt.in_out_degrees,
        smp.world.in_out_degrees
    )
