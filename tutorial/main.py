"""Example usage of the uclasm package for finding subgraph isomorphisms."""

import uclasm

tmplt = uclasm.load_edgelist("template.csv",
                             file_source_col="Source",
                             file_target_col="Target",
                             file_channel_col="eType")

world = uclasm.load_edgelist("world.csv",
                             file_source_col="Source",
                             file_target_col="Target",
                             file_channel_col="eType")

smp = uclasm.MatchingProblem(tmplt, world)

uclasm.matching.local_cost_bound.nodewise(smp)

print(smp)
