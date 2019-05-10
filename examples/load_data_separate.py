import sys
sys.path.append("..")

import uclasm

nodelist = uclasm.load_nodelist(
    "example_data_files/example_nodelist.csv",
    node_col=0,
    label_col=1,
    header=0)

_, channels, adjs = \
    uclasm.load_edgelist(
        "example_data_files/example_edgelist.csv",
        nodelist=nodelist,
        src_col=0,
        dst_col=1,
        channel_col=None,
        header=0)

nodes = nodelist.node
labels = nodelist.label

# Use the same graph data for both template and world graphs
tmplt = uclasm.Graph(nodes, channels, adjs, labels=labels)
world = uclasm.Graph(nodes, channels, adjs, labels=labels)
