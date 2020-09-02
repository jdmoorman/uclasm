import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix

# TODO: make channel column optional

def edgelist_to_adjs(edgelist, nodelist=None):
    edgecounts = edgelist.groupby(by=edgelist.columns.tolist(),
                                  as_index=False).size().reset_index(name="count")

    if "channel" in edgelist.columns:
        channels = edgecounts.channel.unique()

    if nodelist is None:
        nodes = pd.concat([edgecounts.src, edgecounts.dst]).unique()
        nodelist = pd.DataFrame({"node": nodes, "label": None})

    nodes = nodelist.node

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    edgecounts[["src", "dst"]] = edgecounts[["src", "dst"]].applymap(node_to_idx.get)

    if "channel" in edgelist.columns:
        adjs = []
        for channel in channels:
            ch_ec = edgecounts[edgecounts.channel==channel]
            adjs.append(csr_matrix((ch_ec["count"], (ch_ec["src"], ch_ec["dst"])),
                        shape=(len(node_to_idx), len(node_to_idx))))
    else:
        # [None] is used since there are no channels
        channels = [None]
        # Short alias for edgecounts df for ease of typing
        ec = edgecounts
        adjs = [csr_matrix((ec["count"], (ec["src"], ec["dst"])),
                shape=(len(node_to_idx), len(node_to_idx)))]

    return nodelist, channels, adjs

def load_combo(filepath, *,
               node_vs_edge_col=0,
               node_str="v",
               node_col=1,
               label_col=None,
               src_col=1,
               dst_col=2,
               channel_col=3,
               **kwargs):

    if label_col is None:
        usecols = [node_vs_edge_col, node_col]
        names = ["vs", "node"]
    else:
        usecols = [node_vs_edge_col, node_col, label_col]
        names = ["vs", "node", "label"]

    nodelist = pd.read_csv(filepath,
                           usecols=usecols,
                           names=names,
                           engine='python',
                           **kwargs)

    # Get rid of the "vs" column
    nodelist = nodelist[nodelist.vs == node_str][names[1:]]

    if channel_col is None:
        usecols = [node_vs_edge_col, src_col, dst_col]
        names = ["vs", "src", "dst"]
    else:
        usecols = [node_vs_edge_col, src_col, dst_col, channel_col]
        names = ["vs", "src", "dst", "channel"]

    edgelist = pd.read_csv(filepath,
                           usecols=usecols,
                           names=names,
                           engine='python',
                           **kwargs)
    edgelist = edgelist[edgelist.vs != node_str][names[1:]]

    return edgelist_to_adjs(edgelist, nodelist)

def load_nodelist(filepath, *,
                  node_col=0,
                  label_col=None,
                  **kwargs):
    if label_col is None:
        usecols = [node_col]
        names = ["node"]
    else:
        usecols = [node_col, label_col]
        names = ["node", "label"]

    return pd.read_csv(filepath,
                       usecols=usecols,
                       names=names,
                       engine='python',
                       **kwargs)

def load_edgelist(filepath, *,
                  nodelist=None,
                  src_col=0,
                  dst_col=1,
                  channel_col=2,
                  **kwargs):
    if channel_col is None:
        usecols = [src_col, dst_col]
        names = ["src", "dst"]
    else:
        usecols = [src_col, dst_col, channel_col]
        names = ["src", "dst", "channel"]

    edgelist = pd.read_csv(filepath,
                           usecols=usecols,
                           names=names,
                           engine='python',
                           **kwargs)

    return edgelist_to_adjs(edgelist, nodelist)
