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

nodes = nodelist.node
labels = nodelist.label

# Use the same graph data for both template and world graphs
tmplt = uclasm.Graph(nodes, channels, adjs, labels=labels)
world = uclasm.Graph(nodes, channels, adjs, labels=labels)
