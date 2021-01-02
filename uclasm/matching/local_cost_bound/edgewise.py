"""Provide a function for bounding node assignment costs with edgewise info."""
import numpy as np
import pandas as pd
import numba
import os
import tqdm
import time

def iter_adj_pairs(tmplt, world):
    """Generator for pairs of adjacency matrices.

    Each pair of adjacency matrices corresponds to the same channel in both
    the template and the world.

    Parameters
    ----------
    tmplt : Graph
        Template graph to be matched.
    world : Graph
        World graph to be searched.

    Yields
    -------
    (spmatrix, spmatrix)
        A tuple of sparse adjacency matrices
    """
    for channel, tmplt_adj in tmplt.ch_to_adj.items():
        world_adj = world.ch_to_adj[channel]
        yield (tmplt_adj, world_adj)
        yield (tmplt_adj.T, world_adj.T)

def get_src_dst_weights(smp, src_idx, dst_idx):
    """ Returns a tuple of src_weight, dst_weight indicating the weighting for
    edge costs to node costs. Weights sum to 2, as they will later be divided by
    2 in from_local_bounds.
    """
    if isinstance(src_idx, list) or isinstance(dst_idx, list):
        if len(src_idx) == 1:
            return (2, 0)
        elif len(dst_idx) == 1:
            return (0, 2)
    if not smp.use_monotone:
        if hasattr(smp, "next_tmplt_idx") and smp.next_tmplt_idx in [src_idx, dst_idx]:
            if src_idx == smp.next_tmplt_idx:
                return (2, 0)
            elif dst_idx == smp.next_tmplt_idx:
                return (0, 2)
        else:
            assigned_tmplt_idxs = smp.assigned_tmplt_idxs
            if src_idx in assigned_tmplt_idxs and dst_idx not in assigned_tmplt_idxs:
                return (0, 2)
            elif dst_idx in assigned_tmplt_idxs and src_idx not in assigned_tmplt_idxs:
                return (2, 0)
            else:
                return (1, 1)
    else:
        return (1, 1)

def get_edgelist_iterator(edgelist, src_col, dst_col, attr_keys, node_as_str=True):
    n_edges = len(edgelist.index)
    srcs = edgelist[src_col]
    dsts = edgelist[dst_col]
    if node_as_str:
        srcs = srcs.astype(str)
        dsts = dsts.astype(str)
    attr_cols = [edgelist[key] for key in attr_keys]
    return zip(range(n_edges), srcs, dsts, *attr_cols)

from numba import float64, int64, void

@numba.njit(void(float64[:,:], int64, int64[:], float64[:]))
def set_assignment_costs(assignment_costs, tmplt_idx, cand_idxs, attr_costs):
    for cand_idx, attr_cost in zip(cand_idxs, attr_costs):
        if attr_cost < assignment_costs[tmplt_idx, cand_idx]:
            assignment_costs[tmplt_idx, cand_idx] = attr_cost

def get_edge_to_unique_attr(edgelist, src_col, dst_col):
    """Get a map from edge indexes to unique attribute indexes.

    Parameters
    ----------
    edgelist : DataFrame
        Each row of the dataframe should contain the source, destination, and attributes of an edge.
    src_col : str
        Column name for the source of each edge in `edgelist`.
    dst_col : str
        Column name for the destination of each edge in `edgelist`.

    Returns
    -------
    edge_to_attr_idx : dict(int, int)
        Map edge indexes to the indexes of the attributes in a uniquified list of attributes
    unique_attrs : array_like(str)
        The unique attributes for the edges of `graph`
    """
    attr_names = [a for a in edgelist.columns if a not in [src_col, dst_col, 'id', 'template_id']]
    attrs = edgelist[attr_names].to_numpy()
    # attrs1d = [str(attr_row) for attr_row in attrs]
    str_array = np.frompyfunc(str, 1, 1)
    _, index, inverse = np.unique(str_array(attrs).astype(str), axis=0, return_index=True, return_inverse=True)
    unique_attrs = attrs[index]
    # Do we need to turn unique_attrs back into a dataframe?

    return unique_attrs, inverse

def edgewise_no_attrs(smp, changed_cands=None):
    """Compute edgewise costs in the case where no attribute distance function
    is provided.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute edgewise cost bounds.
    changed_cands : ndarray(bool)
        Boolean array indicating which template nodes have candidates that have
        changed since the last run of the edgewise filter. Only these nodes and
        their neighboring template nodes have to be reevaluated.
    """
    new_local_costs = np.zeros(smp.shape)
    candidates = smp.candidates()
    for src_idx, dst_idx in smp.tmplt.nbr_idx_pairs:
        if changed_cands is not None:
            # If neither the source nor destination has changed, there is no
            # point in filtering on this pair of nodes
            if not (changed_cands[src_idx] or changed_cands[dst_idx]):
                continue

        # get indicators of candidate nodes in the world adjacency matrices
        src_is_cand = candidates[src_idx]
        dst_is_cand = candidates[dst_idx]
        if ~np.any(src_is_cand) or ~np.any(dst_is_cand):
            print("No candidates for given nodes, skipping edge")
            continue

        # This sparse matrix stores the number of supported template edges
        # between each pair of candidates for src and dst
        # i.e. the number of template edges between src and dst that also exist
        # between their candidates in the world
        supported_edges = None

        # Number of total edges in the template between src and dst
        total_tmplt_edges = 0
        for tmplt_adj, world_adj in iter_adj_pairs(smp.tmplt, smp.world):
            tmplt_adj_val = tmplt_adj[src_idx, dst_idx]
            total_tmplt_edges += tmplt_adj_val

            # if there are no edges in this channel of the template, skip it
            if tmplt_adj_val == 0:
                continue

            # sub adjacency matrix corresponding to edges from the source
            # candidates to the destination candidates
            world_sub_adj = world_adj[:, dst_is_cand][src_is_cand, :]

            # Edges are supported up to the number of edges in the template
            if supported_edges is None:
                supported_edges = world_sub_adj.minimum(tmplt_adj_val)
            else:
                supported_edges += world_sub_adj.minimum(tmplt_adj_val)

        src_support = supported_edges.max(axis=1)
        src_least_cost = total_tmplt_edges - src_support.A

        # Different algorithm from REU
        # Main idea: assigning u' to u and v' to v causes cost for u to increase
        # based on minimum between cost of v and missing edges between u and v
        # src_least_cost = np.maximum(total_tmplt_edges - supported_edges.A,
        #                             local_costs[dst_idx][dst_is_cand]).min(axis=1)

        src_least_cost = np.array(src_least_cost).flatten()
        # Update the local cost bound
        new_local_costs[src_idx][src_is_cand] += src_least_cost

        if src_idx != dst_idx:
            dst_support = supported_edges.max(axis=0)
            dst_least_cost = total_tmplt_edges - dst_support.A
            dst_least_cost = np.array(dst_least_cost).flatten()
            new_local_costs[dst_idx][dst_is_cand] += dst_least_cost
    return new_local_costs

def generate_edgewise_cost_cache(smp, cache_by_unique_attrs=True):
    """Generate a cache of edgewise costs for later use.

    Parameters
    ----------
    cache_by_unique_attrs : bool
        Cache the edgewise costs between edges with unique attributes, and
        generate the mapping from edges to unique attribute indices.
    """
    if cache_by_unique_attrs:
        print('Calculating edge to unique attr map')
        start_time = time.time()
        if 'importance' not in smp.tmplt.edgelist.columns:
            # Template edges with the same attributes could still be different if they have different importances
            # For now, prevent the code from removing them as duplicates
            tmplt_attr_keys = [attr for attr in smp.tmplt.edgelist.columns if attr not in ['id', 'template_id']]
            tmplt_unique_attrs, tmplt_edge_to_attr_idx = get_edge_to_unique_attr(smp.tmplt.edgelist, None, None)
        else:
            tmplt_unique_attrs, tmplt_edge_to_attr_idx = get_edge_to_unique_attr(smp.tmplt.edgelist, smp.tmplt.source_col, smp.tmplt.target_col)
        world_unique_attrs, world_edge_to_attr_idx = get_edge_to_unique_attr(smp.world.edgelist, smp.world.source_col, smp.world.target_col)
        print('Edge to unique attr map calculated in {} seconds'.format(time.time()-start_time))

        smp.tmplt_edge_to_attr_idx = np.array(tmplt_edge_to_attr_idx)
        smp.world_edge_to_attr_idx = np.array(world_edge_to_attr_idx)

        smp._edgewise_costs_cache = np.zeros((len(tmplt_unique_attrs), len(world_unique_attrs)))
        pbar = tqdm.tqdm(total=len(tmplt_unique_attrs), position=0, leave=True, ascii=True)

        for tmplt_unique_idx, tmplt_attrs in enumerate(tmplt_unique_attrs):
            tmplt_attrs_dict = dict(zip(tmplt_attr_keys, tmplt_attrs))
            if 'importance' in tmplt_attr_keys:
                edge_key = None
            else:
                # Retrieve the source and destination, and reconstruct the edge key
                edge_key = (str(tmplt_attrs_dict[smp.tmplt.source_col]), str(tmplt_attrs_dict[smp.tmplt.target_col]))
                del tmplt_attrs_dict[smp.tmplt.source_col]
                del tmplt_attrs_dict[smp.tmplt.target_col]

            src_col_world = smp.world.source_col
            dst_col_world = smp.world.target_col
            cand_attr_keys = [attr for attr in smp.world.edgelist.columns if attr not in [src_col_world, dst_col_world, 'id']]
            for world_unique_idx, cand_attrs in enumerate(world_unique_attrs):
                cand_attrs_dict = dict(zip(cand_attr_keys, cand_attrs))
                if 'importance' in tmplt_attr_keys:
                    importance = tmplt_attrs_dict['importance']
                else:
                    importance = None
                attr_cost = smp.edge_attr_fn(edge_key, None, tmplt_attrs_dict, cand_attrs_dict, importance_value=importance)
                smp._edgewise_costs_cache[tmplt_unique_idx, world_unique_idx] = attr_cost
            pbar.update(1)
        pbar.close()

    else:
        n_tmplt_edges = len(smp.tmplt.edgelist.index)
        n_world_edges = len(smp.world.edgelist.index)
        smp._edgewise_costs_cache = np.zeros((n_tmplt_edges, n_world_edges))
        pbar = tqdm.tqdm(total=len(smp.tmplt.edgelist.index), position=0, leave=True, ascii=True)
        for tmplt_edge_idx, src_node, dst_node, *tmplt_attrs in get_edgelist_iterator(smp.tmplt.edgelist, src_col, dst_col, tmplt_attr_keys):
            tmplt_attrs_dict = dict(zip(tmplt_attr_keys, tmplt_attrs))

            src_col_world = smp.world.source_col
            dst_col_world = smp.world.target_col
            cand_attr_keys = [attr for attr in smp.world.edgelist.columns if attr not in [src_col_world, dst_col_world]]
            for world_edge_idx, src_cand, dst_cand, *cand_attrs in get_edgelist_iterator(smp.world.edgelist, src_col_world, dst_col_world, cand_attr_keys):
                cand_attrs_dict = dict(zip(cand_attr_keys, cand_attrs))
                if 'importance' in tmplt_attr_keys:
                    attr_cost = smp.edge_attr_fn((src_node, dst_node), (src_cand, dst_cand), tmplt_attrs_dict, cand_attrs_dict, importance_value=tmplt_attrs_dict['importance'])
                else:
                    attr_cost = smp.edge_attr_fn((src_node, dst_node), (src_cand, dst_cand), tmplt_attrs_dict, cand_attrs_dict)
                smp._edgewise_costs_cache[tmplt_edge_idx, world_edge_idx] = attr_cost
            pbar.update(1)
        pbar.close()
    if smp.cache_path is not None:
        np.save(os.path.join(smp.cache_path, "edgewise_costs_cache.npy"), smp._edgewise_costs_cache)
        if cache_by_unique_attrs:
            np.save(os.path.join(smp.cache_path, "tmplt_edge_to_attr_idx.npy"), smp.tmplt_edge_to_attr_idx)
            np.save(os.path.join(smp.cache_path, "world_edge_to_attr_idx.npy"), smp.world_edge_to_attr_idx)
        try:
            os.chmod(smp.cache_path, 0o770)
        except:
            pass
        print("Edge-to-edge costs saved to cache")

def verify_edgewise_cost_cache(smp, cache_by_unique_attrs=True):
    """Check that the edgewise cost cache was successfully loaded and generate
    edge to attr maps if none are found.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute edgewise cost bounds.
    """
    n_tmplt_edges = len(smp.tmplt.edgelist.index)
    n_world_edges = len(smp.world.edgelist.index)
    if cache_by_unique_attrs:
        if not hasattr(smp, 'tmplt_edge_to_attr_idx'):
            try:
                smp.tmplt_edge_to_attr_idx = np.load(os.path.join(smp.cache_path, "tmplt_edge_to_attr_idx.npy"))
                smp.world_edge_to_attr_idx = np.load(os.path.join(smp.cache_path, "world_edge_to_attr_idx.npy"))
                print('Edge to attr maps loaded from cache')
            except:
                print('Edge to attr cache not found')
                print('Calculating edge to unique attr map')
                start_time = time.time()
                if 'importance' not in smp.tmplt.edgelist.columns:
                    tmplt_unique_attrs, tmplt_edge_to_attr_idx = get_edge_to_unique_attr(smp.tmplt.edgelist, None, None)
                else:
                    tmplt_unique_attrs, tmplt_edge_to_attr_idx = get_edge_to_unique_attr(smp.tmplt.edgelist, smp.tmplt.source_col, smp.tmplt.target_col)
                world_unique_attrs, world_edge_to_attr_idx = get_edge_to_unique_attr(smp.world.edgelist, smp.world.source_col, smp.world.target_col)
                smp.tmplt_edge_to_attr_idx = tmplt_edge_to_attr_idx
                smp.world_edge_to_attr_idx = world_edge_to_attr_idx
                print('Edge to unique attr map calculated in {} seconds'.format(time.time()-start_time))
        if len(smp.tmplt_edge_to_attr_idx) != n_tmplt_edges or len(smp.world_edge_to_attr_idx) != n_world_edges:
            raise Exception("Edgewise costs cache not properly computed!")
        # TODO: implement a more effective check here
    else:
        if smp._edgewise_costs_cache.shape != (n_tmplt_edges, n_world_edges):
            raise Exception("Edgewise costs cache not properly computed!")

def edgewise_local_costs(smp, changed_cands=None, use_cost_cache=True,
                         cache_by_unique_attrs=True):
    """Compute edge disagreements between candidates.

    Computes a lower bound on the local cost of assignment by iterating
    over template edges and comparing candidates for the endpoints.
    The lower bound for an assignment (u, u') is the sum over all neighbors
    v of u of the minimum number of missing edges between (u', v') over
    all v' where v' is a candidate for v.

    TODO: Cite paper from REU.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute edgewise cost bounds.
    changed_cands : ndarray(bool)
        Boolean array indicating which template nodes have candidates that have
        changed since the last run of the edgewise filter. Only these nodes and
        their neighboring template nodes have to be reevaluated.
    """

    if smp.edge_attr_fn is None:
        return edgewise_no_attrs(smp, changed_cands=changed_cands)
    else:
        new_local_costs = np.zeros(smp.shape)
        candidates = smp.candidates()
        # Iterate over template edges and consider best matches for world edges
        if use_cost_cache:
            # Edge index -> unique attribute idx
            # Avoid recalculating edge distance for identical attribute tuples
            src_col = smp.tmplt.source_col
            dst_col = smp.tmplt.target_col
            tmplt_attr_keys = [attr for attr in smp.tmplt.edgelist.columns if attr not in [src_col, dst_col, 'id', 'template_id']]

            n_tmplt_edges = len(smp.tmplt.edgelist.index)
            n_world_edges = len(smp.world.edgelist.index)
            if smp._edgewise_costs_cache is None:
                generate_edgewise_cost_cache(smp, cache_by_unique_attrs=cache_by_unique_attrs)
            else:
                verify_edgewise_cost_cache(smp, cache_by_unique_attrs=cache_by_unique_attrs)

            for tmplt_edge_idx, src_node, dst_node, *tmplt_attrs in get_edgelist_iterator(smp.tmplt.edgelist, src_col, dst_col, tmplt_attr_keys, node_as_str=False):
                tmplt_attrs_dict = dict(zip(tmplt_attr_keys, tmplt_attrs))
                if isinstance(src_node, list) or isinstance(dst_node, list):
                    # Handle templates with multiple alternatives
                    if len(src_node) > 1 and len(dst_node) > 1:
                        raise Exception("Edgewise cost bound cannot handle template edges with both multiple sources and multiple destinations.")
                    elif len(src_node) == 1 and len(dst_node) == 1:
                        src_node = src_node[0]
                        dst_node = dst_node[0]
                        src_idx = smp.tmplt.node_idxs[src_node]
                        dst_idx = smp.tmplt.node_idxs[dst_node]
                        src_node, dst_node = str(src_node), str(dst_node)
                    else:
                        src_idx = [smp.tmplt.node_idxs[src_node_i] for src_node_i in src_node]
                        dst_idx = [smp.tmplt.node_idxs[dst_node_i] for dst_node_i in dst_node]
                else:
                    # Get candidates for src and dst
                    src_idx = smp.tmplt.node_idxs[src_node]
                    dst_idx = smp.tmplt.node_idxs[dst_node]
                    src_node, dst_node = str(src_node), str(dst_node)
                # Matrix of costs of assigning template node src_idx and dst_idx
                # to candidates row_idx and col_idx
                assignment_costs = np.zeros(smp.shape)
                if 'importance' in tmplt_attr_keys:
                    missing_edge_cost = smp.missing_edge_cost_fn((src_node, dst_node), tmplt_attrs_dict['importance'])
                else:
                    missing_edge_cost = smp.missing_edge_cost_fn((src_node, dst_node))
                # Put the weight of assignments on the unassigned nodes, when possible
                # Only works if monotone is disabled
                src_weight, dst_weight = get_src_dst_weights(smp, src_idx, dst_idx)
                if src_weight > 0:
                    assignment_costs[src_idx, :] = src_weight * missing_edge_cost
                if dst_weight > 0:
                    assignment_costs[dst_idx, :] = dst_weight * missing_edge_cost

                # TODO: add some data to the graph classes to store the node indexes
                # of the source and destination of each edge. You can then use this
                # to efficiently get your masks by:
                # >>> candidates[src_idx, smp.world.src_idxs]
                world_edge_src_idxs = smp.world.edge_src_idxs
                if isinstance(src_idx, list):
                    cand_edge_src_mask = np.sum(candidates[src_idx, :][:, world_edge_src_idxs], axis=0)
                else:
                    cand_edge_src_mask = candidates[src_idx, world_edge_src_idxs]
                world_edge_dst_idxs = smp.world.edge_dst_idxs
                if isinstance(dst_idx, list):
                    cand_edge_dst_mask = np.sum(candidates[dst_idx, :][:, world_edge_dst_idxs], axis=0)
                else:
                    cand_edge_dst_mask = candidates[dst_idx, world_edge_dst_idxs]
                cand_edge_mask = np.logical_and(cand_edge_src_mask, cand_edge_dst_mask)
                if np.any(cand_edge_mask):
                    src_cand_idxs = world_edge_src_idxs[cand_edge_mask]
                    dst_cand_idxs = world_edge_dst_idxs[cand_edge_mask]
                    # Put the weight of assignments on the unassigned nodes, when possible
                    # Only works if monotone is disabled
                    if cache_by_unique_attrs:
                        # cand_edge_idxs = np.arange(n_world_edges)[cand_edge_mask]
                        # cand_attr_idxs = [smp.world_edge_to_attr_idx[cand_edge_idx] for cand_edge_idx in cand_edge_idxs]
                        cand_attr_idxs = smp.world_edge_to_attr_idx[cand_edge_mask]
                        attr_costs = smp._edgewise_costs_cache[smp.tmplt_edge_to_attr_idx[tmplt_edge_idx], cand_attr_idxs]
                    else:
                        attr_costs = smp._edgewise_costs_cache[tmplt_edge_idx, cand_edge_mask]
                    if src_weight > 0:
                        set_assignment_costs(assignment_costs, src_idx, src_cand_idxs, src_weight * attr_costs)
                    if dst_weight > 0:
                        set_assignment_costs(assignment_costs, dst_idx, dst_cand_idxs, dst_weight * attr_costs)
                new_local_costs += assignment_costs

            if hasattr(smp.tmplt, 'time_constraints'):
                add_time_costs(smp, candidates, new_local_costs)

            if hasattr(smp.tmplt, 'geo_constraints'):
                add_geo_costs(smp, candidates, new_local_costs)

            return new_local_costs
        else:
            src_col = smp.tmplt.source_col
            dst_col = smp.tmplt.target_col
            tmplt_edgelist = smp.tmplt.edgelist
            tmplt_attr_keys = [attr for attr in tmplt_edgelist.columns if attr not in [src_col, dst_col]]
            for tmplt_edge_idx, src_node, dst_node, *tmplt_attrs in get_edgelist_iterator(tmplt_edgelist, src_col, dst_col, tmplt_attr_keys, node_as_str=False):
                tmplt_attrs_dict = dict(zip(tmplt_attr_keys, tmplt_attrs))
                # Get candidates for src and dst
                src_idx = smp.tmplt.node_idxs[src_node]
                dst_idx = smp.tmplt.node_idxs[dst_node]
                src_node, dst_node = str(src_node), str(dst_node)
                # Matrix of costs of assigning template node src_idx and dst_idx
                # to candidates row_idx and col_idx
                assignment_costs = np.zeros(smp.shape)
                missing_edge_cost = smp.missing_edge_cost_fn((src_node, dst_node))
                # Put the weight of assignments on the unassigned nodes, when possible
                # Only works if monotone is disabled
                src_weight, dst_weight = get_src_dst_weights(smp, src_idx, dst_idx)
                if src_weight > 0:
                    assignment_costs[src_idx, :] = src_weight * missing_edge_cost
                if dst_weight > 0:
                    assignment_costs[dst_idx, :] = dst_weight * missing_edge_cost

                # TODO: add some data to the graph classes to store the node indexes
                # of the source and destination of each edge. You can then use this
                # to efficiently get your masks by:
                # >>> candidates[src_idx, smp.world.src_idxs]
                world_edge_src_idxs = smp.world.edge_src_idxs
                cand_edge_src_mask = candidates[src_idx, world_edge_src_idxs]
                world_edge_dst_idxs = smp.world.edge_dst_idxs
                cand_edge_dst_mask = candidates[dst_idx, world_edge_dst_idxs]
                cand_edge_mask = np.logical_and(cand_edge_src_mask, cand_edge_dst_mask)
                cand_edgelist = smp.world.edgelist[cand_edge_mask]
                src_cand_idxs = world_edge_src_idxs[cand_edge_mask]
                dst_cand_idxs = world_edge_dst_idxs[cand_edge_mask]

                src_col_world = smp.world.source_col
                dst_col_world = smp.world.target_col
                cand_attr_keys = [attr for attr in cand_edgelist.columns if attr not in [src_col_world, dst_col_world]]
                src_cands = cand_edgelist[src_col].astype(str)
                dst_cands = cand_edgelist[dst_col].astype(str)
                attr_cols = [cand_edgelist[key] for key in cand_attr_keys]

                for src_cand, dst_cand, src_cand_idx, dst_cand_idx, *cand_attrs in zip(src_cands, dst_cands, src_cand_idxs, dst_cand_idxs, *attr_cols):
                    # src_cand_idx = smp.world.node_idxs[src_cand]
                    # dst_cand_idx = smp.world.node_idxs[dst_cand]
                    # src_cand, dst_cand = str(src_cand), str(dst_cand)
                    cand_attrs_dict = dict(zip(cand_attr_keys, cand_attrs))
                    attr_cost = smp.edge_attr_fn((src_node, dst_node), (src_cand, dst_cand), tmplt_attrs_dict, cand_attrs_dict)

                    # Put the weight of assignments on the unassigned nodes, when possible
                    # Only works if monotone is disabled
                    if src_weight > 0:
                        assignment_costs[src_idx, src_cand_idx] = min(assignment_costs[src_idx, src_cand_idx], src_weight * attr_cost)
                    if dst_weight > 0:
                        assignment_costs[dst_idx, dst_cand_idx] = min(assignment_costs[dst_idx, dst_cand_idx], dst_weight * attr_cost)
                new_local_costs += assignment_costs

    return new_local_costs

def edgewise(smp, changed_cands=None):
    """Bound local assignment costs by edge disagreements between candidates.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute edgewise cost bounds.
    """
    smp.local_costs = edgewise_local_costs(smp, changed_cands)

def add_time_costs(smp, candidates, local_costs):
    """Add costs associated with time constraints

    Parameters
    ----------
    smp: MatchingProblem
        A subgraph matching problem on which to compute edgewise cost bounds.
    candidates: array
        The precomputed array of candidates. Avoids recomputing candidates.
    local_costs: array
        Array of local costs to add the time costs to.
    """
    import datetime
    for time_constraint in smp.tmplt.time_constraints:
        node1_idx = smp.tmplt.node_idxs[time_constraint["node1"]]
        node2_idx = smp.tmplt.node_idxs[time_constraint["node2"]]
        importance = time_constraint["importance"]
        cand1_times = smp.world.nodelist["start"][candidates[node1_idx]].to_numpy()
        cand2_times = smp.world.nodelist["start"][candidates[node2_idx]].to_numpy()

        # Flat inverse importance for missing information
        cand1_costs = np.ones(cand1_times.shape) / importance
        cand2_costs = np.ones(cand2_times.shape) / importance

        cand1_nonnat_mask = np.logical_not(np.isnat(cand1_times))
        cand2_nonnat_mask = np.logical_not(np.isnat(cand2_times))
        cand1_times = cand1_times[cand1_nonnat_mask]
        cand2_times = cand2_times[cand2_nonnat_mask]
        cand1_nonnat_costs = np.ones(cand1_times.shape) / importance
        cand2_nonnat_costs = np.ones(cand2_times.shape) / importance

        # TODO: support constraints measured in units other than days
        min_timedelta = np.timedelta64(0, 'D')
        if 'minValue' in time_constraint:
            min_timedelta = np.timedelta64(int(time_constraint["minValue"]), 'D')
        if 'maxValue' in time_constraint:
            max_timedelta = np.timedelta64(int(time_constraint["maxValue"]), 'D')

        if len(cand1_times) == 0:
            pass
        elif len(cand2_times) == 0:
            pass
        else:
            cand1_sorted_idxs = cand1_times.argsort()
            cand2_sorted_idxs = cand2_times.argsort()

            cand2_idx_idx = 0
            cand2_idx = cand2_sorted_idxs[cand2_idx_idx]
            cand2_time = cand2_times[cand2_idx]
            last_valid_cand2_idx_idx = 0
            for cand1_idx_idx, cand1_idx in enumerate(cand1_sorted_idxs):
                cand1_time = cand1_times[cand1_idx]
                while cand2_time - cand1_time < min_timedelta:
                    cand2_idx_idx += 1
                    if cand2_idx_idx >= len(cand2_sorted_idxs):
                        cand2_time = None
                        break
                    cand2_idx = cand2_sorted_idxs[cand2_idx_idx]
                    cand2_time = cand2_times[cand2_idx]
                if cand2_time is not None:
                    if 'maxValue' in time_constraint:
                        if cand2_time - cand1_time > max_timedelta:
                            continue
                    cand1_nonnat_costs[cand1_idx] = 0
                    if 'maxValue' in time_constraint:
                        cand2_nonnat_costs[cand2_idx] = 0
                        temp_cand2_idx_idx = max(last_valid_cand2_idx_idx, cand2_idx_idx) + 1
                        if temp_cand2_idx_idx < len(cand2_sorted_idxs):
                            next_cand2_idx = cand2_sorted_idxs[temp_cand2_idx_idx]
                            next_cand2_time = cand2_times[next_cand2_idx]
                            while next_cand2_time - cand1_time <= max_timedelta:
                                cand2_nonnat_costs[next_cand2_idx] = 0
                                last_valid_cand2_idx_idx = temp_cand2_idx_idx
                                temp_cand2_idx_idx += 1
                                if temp_cand2_idx_idx < len(cand2_sorted_idxs):
                                    next_cand2_idx = cand2_sorted_idxs[temp_cand2_idx_idx]
                                    next_cand2_time = cand2_times[next_cand2_idx]
                                else:
                                    break
                    elif cand1_idx_idx == 0:
                        # With only minimum timedelta, all future cand2_times are valid
                        cand2_nonnat_costs[cand2_sorted_idxs[cand2_idx_idx:]] = 0
                else:
                    break
        cand1_costs[cand1_nonnat_mask] = cand1_nonnat_costs
        cand2_costs[cand2_nonnat_mask] = cand2_nonnat_costs
        node1_weight, node2_weight = get_src_dst_weights(smp, node1_idx, node2_idx)
        if node1_weight > 0:
            local_costs[node1_idx, candidates[node1_idx]] += cand1_costs * node1_weight
        if node2_weight > 0:
            local_costs[node2_idx, candidates[node2_idx]] += cand2_costs * node2_weight

def valid_lat_lng(lat, lng):
    """Checks that the lat and lng values are valid

    Parameters
    ----------
    lat: float
        Latitude value.
    lng: float
        Longitude value.
    """
    return (lat >= -90.0) and (lat <= 90.0) and (lng >= -180.0) and (lng <= 180.0)

from math import sin, cos, sqrt, atan2, radians
def haversine(lat1, lon1, lat2, lon2):
    #Haversine distance

    # approximate radius of earth in meters
    R = 6373.0 * 1000

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def add_geo_costs(smp, candidates, local_costs):
    """Add costs associated with geo constraints

    Parameters
    ----------
    smp: MatchingProblem
        A subgraph matching problem on which to compute edgewise cost bounds.
    candidates: array
        The precomputed array of candidates. Avoids recomputing candidates.
    local_costs: array
        Array of local costs to add the geo costs to.
    """
    from geopy import distance
    for geo_constraint in smp.tmplt.geo_constraints:
        node1_idx = smp.tmplt.node_idxs[geo_constraint["node1"]]
        node2_idx = smp.tmplt.node_idxs[geo_constraint["node2"]]
        importance = geo_constraint["importance"]
        cand1_lats = smp.world.nodelist["latitude"][candidates[node1_idx]]
        cand1_lngs = smp.world.nodelist["longitude"][candidates[node1_idx]]
        cand2_lats = smp.world.nodelist["latitude"][candidates[node2_idx]]
        cand2_lngs = smp.world.nodelist["longitude"][candidates[node2_idx]]

        # Flat inverse importance costs for missing information
        cand1_costs = np.ones(cand1_lats.shape) / importance
        cand2_costs = np.ones(cand2_lats.shape) / importance

        cand1_nonnull_mask = np.logical_and(~cand1_lats.isin(["", "%NA%", "%NULL%"]).to_numpy(),
                                            ~cand1_lngs.isin(["", "%NA%", "%NULL%"]).to_numpy())
        cand2_nonnull_mask = np.logical_and(~cand2_lats.isin(["", "%NA%", "%NULL%"]).to_numpy(),
                                            ~cand2_lngs.isin(["", "%NA%", "%NULL%"]).to_numpy())
        cand1_costs[cand1_nonnull_mask] = 0
        cand2_costs[cand2_nonnull_mask] = 0

        cand1_lats = cand1_lats[cand1_nonnull_mask].astype("float")
        cand1_lngs = cand1_lngs[cand1_nonnull_mask].astype("float")
        cand2_lats = cand2_lats[cand2_nonnull_mask].astype("float")
        cand2_lngs = cand2_lngs[cand2_nonnull_mask].astype("float")

        # Calculate geo costs matrix: memory inefficient, only use for a low number of nodes
        cand1_n_geo_nodes = len(cand1_lats)
        cand2_n_geo_nodes = len(cand2_lats)
        if cand1_n_geo_nodes > 0 and cand2_n_geo_nodes > 0:
            geo_costs = np.zeros((cand1_n_geo_nodes, cand2_n_geo_nodes))
            for cand1_i, cand1_lat, cand1_lng in zip(range(cand1_n_geo_nodes), cand1_lats, cand1_lngs):
                if not valid_lat_lng(cand1_lat, cand1_lng):
                    geo_costs[cand1_i, :] = 1.0/importance
                    continue
                for cand2_i, cand2_lat, cand2_lng in zip(range(cand2_n_geo_nodes), cand2_lats, cand2_lngs):
                    if not valid_lat_lng(cand2_lat, cand2_lng):
                        geo_costs[cand1_i, cand2_i] = 1.0/importance
                        continue

                    # TODO: Support constraints measured in units other than meters
                    geo_distance = distance.distance((cand1_lat, cand1_lng), (cand2_lat, cand2_lng)).meters
                    # geo_distance = haversine(cand1_lat, cand1_lng, cand2_lat, cand2_lng)

                    if geo_distance > geo_constraint["maxValue"] or geo_distance < geo_constraint["minValue"]:
                        geo_costs[cand1_i, cand2_i] = 1.0/importance
            cand1_costs[cand1_nonnull_mask] = np.min(geo_costs, axis=1)
            cand2_costs[cand2_nonnull_mask] = np.min(geo_costs, axis=0)

        node1_weight, node2_weight = get_src_dst_weights(smp, node1_idx, node2_idx)
        if node1_weight > 0:
            local_costs[node1_idx, candidates[node1_idx]] += cand1_costs * node1_weight
        if node2_weight > 0:
            local_costs[node2_idx, candidates[node2_idx]] += cand2_costs * node2_weight
