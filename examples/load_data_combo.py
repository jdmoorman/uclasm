import sys
sys.path.append("..")

import numpy as np
import uclasm

nodelist, channels, adjs = uclasm.load_combo(
    "example_data_files/example_combo.csv",
    node_vs_edge_col=0,
    node_str="v",
    src_col=1,
    dst_col=2,
    channel_col=3,
    node_col=1,
    label_col=2,
    header=0)

# Use the same graph data for both template and world graphs
world_nodes = nodelist.node
tmplt_nodes = nodelist.node

world_adj_mats = adjs
tmplt_adj_mats = adjs

# initial candidate set for template nodes is the full set of world nodes
tmplt = uclasm.Template(world_nodes, tmplt_nodes, channels, tmplt_adj_mats)
world = uclasm.World(world_nodes, channels, world_adj_mats)

# get rid of candidates whose labels don't match
tmplt_labels = np.array(nodelist.label).reshape(-1, 1)
world_labels = np.array(nodelist.label).reshape(-1, 1)
tmplt.is_cand &= tmplt_labels == world_labels.T
