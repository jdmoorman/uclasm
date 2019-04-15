import uclasm
import numpy as np
from scipy import sparse

# First we must generate some data
# The number of edges between nodes will be iid geometric with the chosen params
# A signal will be placed in the top left block of the world graph by addition

n_channels = 2
n_tmplt_nodes = 5
n_world_nodes = 100
tmplt_p = 0.85
world_p = 0.8

# Now generate world and template graphs, inserting a signal in the top left

np.random.seed(0)

channels = list(range(n_channels))

# TODO: allow for nodes of arbitrary dtype

# Note: by design, the template nodes have the same ids as the signal nodes
tmplt_nodes = np.arange(n_world_nodes, n_world_nodes+n_tmplt_nodes)
world_nodes = np.arange(n_world_nodes, 2*n_world_nodes)

tmplt_adj_mats = []
world_adj_mats = []

for channel in channels:
    tmplt_adj = np.random.geometric(tmplt_p, (n_tmplt_nodes, n_tmplt_nodes)) - 1
    world_adj = np.random.geometric(world_p, (n_world_nodes, n_world_nodes)) - 1

    # Embed a signal in the top left block of the world graph
    world_adj[:n_tmplt_nodes, :n_tmplt_nodes] += tmplt_adj

    tmplt_adj_mats.append(sparse.csc_matrix(tmplt_adj))
    world_adj_mats.append(sparse.csc_matrix(world_adj))


# initial candidate set for template nodes is the full set of world nodes
tmplt = uclasm.Template(world_nodes, tmplt_nodes, channels, tmplt_adj_mats)
world = uclasm.World(world_nodes, channels, world_adj_mats)

uclasm.all_filters(tmplt, world, elimination=True, verbose=True)
n_isomorphisms = uclasm.count_isomorphisms(tmplt, world, verbose=False)

print("\nFound", n_isomorphisms, "isomorphisms")
