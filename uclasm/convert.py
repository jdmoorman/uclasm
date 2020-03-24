"""Functions for converting data from one representation to another."""

import pandas as pd


def nodelist_from_edgelist(edgelist):
    """Extract node names from sources and targets in an edgelist.

    Parameters
    ----------
    edgelist : DataFrame
        A DataFrame each row of which represents an edge. The edgelist should
        minimally have columns matching source_col and target_col.

    Returns
    -------
    DataFrame
        A DataFrame with a single column whose name is taken from node_col. The
        entries of this column are each the name of a node.
    """
    from uclasm.graph import Graph

    sources = edgelist[[Graph.source_col]]
    sources = sources.rename(columns={Graph.source_col: Graph.node_col})

    targets = edgelist[[Graph.target_col]]
    targets = targets.rename(columns={Graph.target_col: Graph.node_col})

    # TODO: What if this is a dask dataframe?
    nodelist = pd.concat([sources, targets])
    nodelist.drop_duplicates(inplace=True, ignore_index=True)

    return nodelist
