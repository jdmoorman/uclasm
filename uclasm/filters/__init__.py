from .label_filter import label_filter
from .stats_filter import stats_filter
from .topology_filter import topology_filter
from .neighborhood_filter import neighborhood_filter
from .permutation_filter import permutation_filter
from .run_filters import run_filters

# These are the most commonly used filters
cheap_filters = [stats_filter, topology_filter]

# This needs to be imported after cheap_filters is defined since it relies
# on cheap_filters
from .elimination_filter import elimination_filter

# Elimination filter is also frequently used
all_filters = cheap_filters + [elimination_filter]
