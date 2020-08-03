import uclasm
import uclasm.utils.data_structures as ds
import networkx as nx

from run_solvers import *


tmplt = ds.from_networkx_graph(nx.erdos_renyi_graph(10, 0.4))
world = ds.from_networkx_graph(nx.erdos_renyi_graph(20, 0.2))

# print(run_LAD(tmplt, world))
print(run_LAD(tmplt, world, enum=True))
