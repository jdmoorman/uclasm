"""Provide a function for bounding node assignment costs with edgewise info."""
import numpy as np
import pandas as pd
import numba

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

class _cache():
    edgewise_costs_cache = None

def get_src_dst_weights(smp, src_idx, dst_idx):
    """ Returns a tuple of src_weight, dst_weight indicating the weighting for
    edge costs to node costs. Weights sum to 2, as they will later be divided by
    2 in from_local_bounds.
    """
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

def edgewise_local_costs(smp, changed_cands=None, use_cost_cache=True):
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
    new_local_costs = np.zeros(smp.shape)
    candidates = smp.candidates()

    if smp.edge_attr_fn is None:
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
    else:
        # Iterate over template edges and consider best matches for world edges
        if use_cost_cache:
            src_col = smp.tmplt.source_col
            dst_col = smp.tmplt.target_col
            tmplt_attr_keys = [attr for attr in smp.tmplt.edgelist.columns if attr not in [src_col, dst_col]]

            global _cache
            n_tmplt_edges = len(smp.tmplt.edgelist.index)
            n_world_edges = len(smp.world.edgelist.index)
            if _cache.edgewise_costs_cache is None or _cache.edgewise_costs_cache.shape != (n_tmplt_edges, n_world_edges):
                _cache.edgewise_costs_cache = np.zeros((n_tmplt_edges, n_world_edges))

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
                        _cache.edgewise_costs_cache[tmplt_edge_idx, world_edge_idx] = attr_cost

            for tmplt_edge_idx, src_node, dst_node, *tmplt_attrs in get_edgelist_iterator(smp.tmplt.edgelist, src_col, dst_col, tmplt_attr_keys, node_as_str=False):
                tmplt_attrs_dict = dict(zip(tmplt_attr_keys, tmplt_attrs))
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
                cand_edge_src_mask = candidates[src_idx, world_edge_src_idxs]
                world_edge_dst_idxs = smp.world.edge_dst_idxs
                cand_edge_dst_mask = candidates[dst_idx, world_edge_dst_idxs]
                cand_edge_mask = np.logical_and(cand_edge_src_mask, cand_edge_dst_mask)
                if np.any(cand_edge_mask):
                    src_cand_idxs = world_edge_src_idxs[cand_edge_mask]
                    dst_cand_idxs = world_edge_dst_idxs[cand_edge_mask]
                    # Put the weight of assignments on the unassigned nodes, when possible
                    # Only works if monotone is disabled
                    attr_costs = _cache.edgewise_costs_cache[tmplt_edge_idx, cand_edge_mask]
                    if src_weight > 0:
                        set_assignment_costs(assignment_costs, src_idx, src_cand_idxs, src_weight * attr_costs)
                    if dst_weight > 0:
                        set_assignment_costs(assignment_costs, dst_idx, dst_cand_idxs, dst_weight * attr_costs)
                new_local_costs += assignment_costs
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
