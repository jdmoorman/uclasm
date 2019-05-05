import sys
sys.path.append("..")

import uclasm

from load_data_combo import tmplt, world

uclasm.run_filters(tmplt, world, uclasm.all_filters, verbose=True)
n_isomorphisms = uclasm.count_isomorphisms(tmplt, world, verbose=False)

print("\nFound", n_isomorphisms, "isomorphisms")
