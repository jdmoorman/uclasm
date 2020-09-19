"""Utility functions and classes for search"""
import numpy as np

from .. import global_cost_bound
from .. import local_cost_bound

class State:
    """A state for the greedy search algorithm.
    Attributes
    ----------
    matching : tuple
        Tuple representation of the current matching.
    cost: float
        Estimated cost of the matching.
    """
    def __init__(self):
        self.matching = None
        self.cost = float("inf")

    def __lt__(self, other):
        # TODO: Is this function the source of your sorting related time expenditures?
        if len(self.matching) != len(other.matching):
            return len(self.matching) > len(other.matching)
        return self.cost < other.cost

    def __str__(self):
        return str(self.matching) + ": " + str(self.cost)

def tuple_from_dict(dict):
    """Turns a dict into a representative sorted tuple of 2-tuples.
    Parameters
    ----------
    dict: dict
        A dictionary.
    Returns
    -------
    tuple
        A unique tuple representation of the dictionary as a sequence of
        2-tuples of key-value pairs, sorted by the dictionary key.
    """
    return tuple(sorted(dict.items()))

def dict_from_tuple(tuple):
    """Turns a tuple of 2-tuples into a dict.
    Parameters
    ----------
    tuple: tuple
        A unique tuple representation of the dictionary as a sequence of
        2-tuples of key-value pairs. Keys must be unique.
    Returns
    -------
    dict
        The equivalent dictionary.
    """
    return dict(tuple)

# This function is now deprecated, use MatchingProblem.enforce_matching instead
def set_fixed_costs(fixed_costs, matching):
    """Set fixed costs to float("inf") to enforce the given matching."""
    mask = np.zeros(fixed_costs.shape, dtype=np.bool)
    mask[[pair[0] for pair in matching],:] = True
    mask[:,[pair[1] for pair in matching]] = True
    mask[tuple(np.array(matching).T)] = False
    fixed_costs[mask] = float("inf")

import tqdm

def add_node_attr_costs(smp, node_attr_fn):
    """Increase the fixed costs to account for difference in node attributes."""
    tmplt_attr_keys = [attr for attr in smp.tmplt.nodelist.columns]
    tmplt_attr_cols = [smp.tmplt.nodelist[key] for key in tmplt_attr_keys]
    tmplt_attrs_zip = zip(*tmplt_attr_cols)
    world_attr_keys = [attr for attr in smp.world.nodelist.columns]
    world_attr_cols = [smp.world.nodelist[key] for key in world_attr_keys]
    world_attrs_zip = zip(*world_attr_cols)
    with tqdm.tqdm(total=smp.tmplt.n_nodes * smp.world.n_nodes, ascii=True) as pbar:
        # for tmplt_idx, tmplt_row in smp.tmplt.nodelist.iterrows():
        for tmplt_idx, tmplt_attrs in enumerate(tmplt_attrs_zip):
            # for world_idx, world_row in smp.world.nodelist.iterrows():
            for world_idx, world_attrs in enumerate(world_attrs_zip):
                pbar.update(1)
                if smp.fixed_costs[tmplt_idx, world_idx] != float("inf"):
                    tmplt_row = dict(zip(tmplt_attr_keys, tmplt_attrs))
                    world_row = dict(zip(world_attr_keys, world_attrs))
                    smp.fixed_costs[tmplt_idx, world_idx] += node_attr_fn(tmplt_row, world_row)

def add_node_attr_costs_identity(smp):
    """Assume node attr fn is the sum of the difference between attributes."""
    world_nodelist_np = np.array(smp.world.nodelist)
    for tmplt_idx, tmplt_row in smp.tmplt.nodelist.iterrows():
        tmplt_row_np = np.array(tmplt_row)
        # Remove first column: node ID which shouldn't be checked
        # Index to remove empty attributes
        nonempty_attrs = tmplt_row[1:] != ""
        smp.fixed_costs[tmplt_idx] += (tmplt_row_np[None, 1:][:,nonempty_attrs] != world_nodelist_np[:,1:][:,nonempty_attrs]).sum(axis=1)

def iterate_to_convergence(smp, reduce_world=True, nodewise=True,
                           edgewise=True, changed_cands=None, verbose=False):
    """Iterates the various cost bounds until the costs converge.
    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem to iterate cost bounds on until convergence
    reduce_world : bool
        Option to reduce the world by removing world nodes that are not
        candidates for any template node.
    changed_cands : np.ndarray(bool)
        Array of boolean values indicating which candidate nodes have changed
        candidates since the last time filters were run.
    verbose : bool
        Flag for verbose output.
    """
    if changed_cands is None:
        changed_cands = np.ones((smp.tmplt.n_nodes,), dtype=np.bool)

    old_candidates = smp.candidates().copy()
    global_cost_bound.from_local_bounds(smp)

    # TODO: Does this break if nodewise changes the candidates?
    while True:
        if nodewise:
            if verbose:
                print(smp)
                print("Running nodewise cost bound")
            local_cost_bound.nodewise(smp)
            global_cost_bound.from_local_bounds(smp)
        if edgewise:
            if verbose:
                print(smp)
                print("Running edgewise cost bound")
            # TODO: does changed_cands work for edgewise?
            # local_cost_bound.edgewise(smp, changed_cands=changed_cands)
            local_cost_bound.edgewise(smp)
            global_cost_bound.from_local_bounds(smp)
        candidates = smp.candidates()
        if ~np.any(candidates):
            break
        changed_cands = np.any(candidates != old_candidates, axis=1)
        if ~np.any(changed_cands):
            break
        if reduce_world:
            smp.reduce_world()
        old_candidates = smp.candidates().copy()
        if smp.match_fixed_costs:
            # Remove non-candidates permanently by setting fixed costs to infinity
            non_cand_mask = np.ones(smp.shape, dtype=np.bool)
            non_cand_mask[old_candidates] = False
            smp.fixed_costs[non_cand_mask] = float("inf")
    if verbose:
        print(smp)
