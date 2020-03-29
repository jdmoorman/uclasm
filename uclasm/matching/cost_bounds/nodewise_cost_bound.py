"""Provide a function for bounding node assignment costs with nodewise info."""

import numpy as np


def nodewise_cost_bound(smp):
    """Compare local features such as degrees between nodes.

    TODO: Cite paper from REU.
    TODO: Describe this function in more detail.
    TODO: Switch features back to a function with optional argument for which
    features to compute.
    TODO: Take candidacy into account when computing features.
    TODO: Bring back reciprocated edges, carefully.
    TODO: Dask parallelization?

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute nodewise cost bounds.
    """
    for idx in range(smp.tmplt.n_nodes):
        tmplt_node_feats = smp.tmplt.features[:, [idx]]
        # TODO: Insert generic loss function between features here
        missing = np.maximum(tmplt_node_feats - smp.world.features, 0)
        smp.update_costs(np.sum(missing, axis=0), indexer=idx)

    # # The implementation above is faster and uses less memory than the
    # # one below.
    # # [n_features, n_tmplt_nodes, n_world_nodes] array of differences
    # feature_diffs = self.tmplt.features[:, :, None] - \
    #                 self.world.features[:, None, :]
    # missing = np.maximum(feature_diffs, 0)
