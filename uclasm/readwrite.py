"""Functions for loading graphs from files and storing them in files."""
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from scipy.sparse import csr_matrix

from .graph import Graph
from .convert import nodelist_from_edgelist
from .utils import apply_index_map_to_cols


# TODO: Make channel column optional
# TODO: Store matching problem results to files?
# TODO: Function for storing graph to file


def load_edgelist(filepath, *,
                  nodelist=None,
                  file_source_col=Graph.source_col,
                  file_target_col=Graph.target_col,
                  file_channel_col=Graph.channel_col):
    """Load an edgelist file into a Graph object.

    TODO: optional argument for putting the edgelist in a pandas dataframe.

    Parameters
    ----------
    filepath : str
        Path to the edgelist file. File should be csv formatted with columns
        corresponding to the source, target, and channel of each edge.
    nodelist : DataFrame, optional
        Nodes of the graph and their attributes. Should have a column named
        by the value of node_col. Extracted from the source and target columns
        of the edgelist if not provided.
    file_source_col : str, optional
        Name of the column in the csv corresponding to the source node.
    file_target_col : str, optional
        Name of the column in the csv corresponding to the target node.
    file_channel_col : str, optional
        Name of the column in the csv corresponding to the edge type.

    Returns
    -------
    Graph
        The graph represented by the edgelist.
    """
    # Using dask rather than pandas for the read allows us to handle large
    # datasets in parallel.
    edgelist = dd.read_csv(filepath, dtype={
        file_source_col: str,
        file_target_col: str,
        file_channel_col: str
    })
    edgelist = edgelist.rename(columns={file_source_col:  Graph.source_col,
                                        file_target_col:  Graph.target_col,
                                        file_channel_col: Graph.channel_col})

    # Count the number of edges between each pair of nodes in each channel.
    by = [Graph.source_col, Graph.target_col, Graph.channel_col]
    with ProgressBar():
        # This ends up being a pandas DataFrame
        edgecounts = edgelist.groupby(by=by).size().reset_index().compute()

    count_col = "Count"

    # Fix column name from dask.
    edgecounts.rename(columns={0: count_col}, inplace=True)

    # Get all the distinct types of edge.
    channels = sorted(edgecounts[Graph.channel_col].unique())

    # Get a node list from the source and target nodes of the edgelist.
    if nodelist is None:
        nodelist = nodelist_from_edgelist(edgecounts)

    n_nodes = len(nodelist)

    # Swap node names for their indices so we can construct matrices.
    nodes = nodelist[Graph.node_col]
    node_cols = [Graph.source_col, Graph.target_col]
    apply_index_map_to_cols(edgecounts, node_cols, nodes)

    # Extract adjacency matrices from the edge counts.
    adjs = []
    for channel in channels:
        ch_edgecounts = edgecounts[edgecounts[Graph.channel_col] == channel]
        ch_counts = ch_edgecounts[count_col]
        ch_sources = ch_edgecounts[Graph.source_col]
        ch_targets = ch_edgecounts[Graph.target_col]

        adjs.append(csr_matrix((ch_counts, (ch_sources, ch_targets)),
                               shape=(n_nodes, n_nodes)))

    return Graph(adjs, channels, nodelist, edgelist)
