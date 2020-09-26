import uclasm

nodelist, channels, adjs = \
    uclasm.load_edgelist(
        "example_data_files/example_edgelist.csv",
        src_col=0,
        dst_col=1,
        channel_col=2,
        header=0)

nodes = nodelist.node
labels = nodelist.label

# Use the same graph data for both template and world graphs
tmplt = uclasm.Graph(nodes, channels, adjs, labels=labels)
world = uclasm.Graph(nodes, channels, adjs, labels=labels)
