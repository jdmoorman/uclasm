import uclasm
import uclasm.utils.data_structures as ds
import networkx as nx

from run_solvers import *


"""Note: Solnon benchmarks do not take isolated nodes into account."""


tmplt = ds.from_networkx_graph(nx.erdos_renyi_graph(5, 0.2))
world = ds.from_networkx_graph(nx.erdos_renyi_graph(10, 0.2))

LAD_output = run_LAD(tmplt, world, enum=True)
candidates, num_isomorphisms, runtime = parse_LAD_output(tmplt, world, LAD_output)

import IPython; IPython.embed()
