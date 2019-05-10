import sys
sys.path.append("..")
import numpy as np

import uclasm

from load_data_combo import tmplt, world

tmplt, world, candidates = uclasm.run_filters(tmplt, world, filters=uclasm.all_filters, verbose=True)
n_isomorphisms = uclasm.count_isomorphisms(tmplt, world, candidates=candidates, verbose=False)

print("\nFound", n_isomorphisms, "isomorphisms")
