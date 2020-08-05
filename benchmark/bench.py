import uclasm
import uclasm.utils.data_structures as ds
import networkx as nx
import time

from run_solvers import *


"""Note: Solnon benchmarks do not take isolated nodes into account."""

def connected_component_subgraphs(G):
        for c in nx.connected_components(G):
            yield G.subgraph(c)

tmplt_G = max(connected_component_subgraphs(nx.erdos_renyi_graph(10, 0.3)), key=len)
world_G = max(connected_component_subgraphs(nx.erdos_renyi_graph(30, 0.2)), key=len)

tmplt = ds.from_networkx_graph(tmplt_G)
world = ds.from_networkx_graph(world_G)

LAD_output = run_LAD(tmplt, world, enum=True)
print(LAD_output)

candidates_LAD, num_isomorphisms, runtime_LAD = parse_LAD_output(tmplt, world, LAD_output)

start_uclasm = time.time()
tmplt, world, candidates_uclasm = uclasm.run_filters(tmplt, world)
runtime_uclasm = time.time()-start_uclasm

assert (np.all(candidates_LAD==candidates_uclasm))

print("LAD takes {}s; UCLASM takes {}s".format(runtime_LAD, runtime_uclasm))

import IPython; IPython.embed()
