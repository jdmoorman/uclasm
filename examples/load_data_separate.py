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
