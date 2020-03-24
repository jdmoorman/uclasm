"""This module provides a class for representing subgraph matching problems."""
import numpy as np
from loguru import logger

from .lsap import constrained_lsap_costs


class MatchingProblem:
    """A class representing any subgraph matching problem, noisy or otherwise.

    TODO: costs -> structural_costs.
    TODO: Switch candidates back to being a property, add a setter for the old
    interface. In the setter, modify the costs to infinity as needed. Don't
    forget to worry about setting slices.
    TODO: describe the class in more detail.
    TODO: optionally accept ground truth map argument.
    TODO: Is it okay to describe the tmplt and world attributes using the same
    descriptions as were used for the corresponding parameters?

    Examples
    --------
    >>> tmplt = uclasm.load_edgelist(template_filepath)
    >>> world = uclasm.load_edgelist(world_filepath)
    >>> smp = uclasm.MatchingProblem(tmplt, world)

    Parameters
    ----------
    tmplt : Graph
        Template graph to be matched.
    world : Graph
        World graph to be searched.
    ground_truth_provided : bool, optional
        A flag indicating whether a signal has been injected into the world
        graph with node identifiers that match those in the template.
    cost_threshold : int, optional
        A subgraph whose cost againt the template exceeds this threshold will
        not be considered a match. It can also be used to eliminate candidates
        from the world graph. A cost of 0 corresponds to an exact match for the
        template, whereas a cost of 1 means that the match may be missing a
        single edge which is present in the template but not in the world.
    candidate_print_limit : int, optional
        When summarizing the candidates of each template node, limit the list
        of candidates to this many.

    Attributes
    ----------
    tmplt : Graph
        Template graph to be matched.
    world : Graph
        World graph to be searched for.
    costs : 2darray
        Each entry of this matrix denotes the minimum local cost of matching
        the template node corresponding to the row to the world node
        corresponding to the column.
    """

    def __init__(self,
                 tmplt, world,
                 ground_truth_provided=True,
                 cost_threshold=0,
                 candidate_print_limit=10):
        self.tmplt = tmplt
        tmplt_channels = set(self.tmplt.channels)
        world_channels = set(world.channels)
        if tmplt_channels != world_channels:
            logger.warning("World channels {} do not appear in template.",
                           world_channels - tmplt_channels)
        world = world.channel_subgraph(self.tmplt.channels)
        self.world = world

        self.costs = np.zeros((tmplt.n_nodes, world.n_nodes))
        self._ground_truth_provided = ground_truth_provided
        self._cost_threshold = cost_threshold
        self._candidate_print_limit = candidate_print_limit

    def candidates(self):
        """Get the matrix of compatibility between template and world nodes.

        World node j is considered to be a candidate for a template node i if
        there exists an assignment from template nodes to world nodes in which
        i is assigned to j whose cost does not exceed the desired threshold.

        Returns
        -------
        2darray
            A boolean matrix where each entry indicates whether the world node
            corresponding to the column is a candidate for the template node
            corresponding to the row.
        """
        # TODO: Avoid recomputing this sum too frequently. Perhaps in the
        # update_costs function we could set a flag if the costs are modified.
        cost_sum = self.costs.sum()

        # Cache the results by checking the sum of the costs against the sum
        # from the last time the candidates were computed. If the sum has
        # changed, the candidates will need to be recomputed.
        if hasattr(self, "_candidates") and hasattr(self, "_cost_sum"):
            if cost_sum == self._cost_sum:
                return self._candidates

        total_costs = constrained_lsap_costs(self.costs)

        self._cost_sum = cost_sum
        self._candidates = total_costs <= self._cost_threshold
        return self._candidates

    def update_costs(self, new_costs, indexer=None):
        """Update the costs with the larger of the old costs and the new.

        Each entry of self.costs is monotonically increasing in time.

        Parameters
        ----------
        new_costs : ndarray
            Costs to update with. Any current costs that are larger than the
            corresponding new costs are kept.
        indexer : ndarray
            The elements of self.costs that are to be updated.
        """
        if indexer is None:
            self.costs = np.maximum(self.costs, new_costs)
        else:
            self.costs[indexer] = np.maximum(self.costs[indexer], new_costs)

    def stats_filter(self):
        """Compare local features such as degrees between nodes.

        TODO: The particular features are important in order to preserve the
        interpretation of self.costs as a lower bound on the local number of
        missing edges under a particular assignment. If double counting occurs
        or features such as the number of reciprocated edges are used, the
        interpretation will be lost. Thus, it could be wise to move the
        features to the MatchingProblem class rather than the Graph class.
        TODO: Describe this function in more detail.
        TODO: This should take candidacy into account when computing features.
        """
        for idx in range(self.tmplt.n_nodes):
            tmplt_node_feats = self.tmplt.features[:, [idx]]
            missing = np.maximum(tmplt_node_feats - self.world.features, 0)
            self.update_costs(np.sum(missing, axis=0), indexer=idx)

        # # The implementation above is faster and uses less memory than the
        # # one below.
        # # [n_features, n_tmplt_nodes, n_world_nodes] array of differences
        # feature_diffs = self.tmplt.features[:, :, None] - \
        #                 self.world.features[:, None, :]
        # missing = np.maximum(feature_diffs, 0)

    # def topology_filter(self):
    #     """Compare edges between each pair of nodes."""
    #     for src_idx, dst_idx in tmplt.nbr_idx_pairs:
    #         if changed_cands is not None:
    #             # If neither the source nor destination has changed, there is no
    #             # point in filtering on this pair of nodes
    #             if not (changed_cands[src_idx] or changed_cands[dst_idx]):
    #                 continue
    #
    #         # get indicators of candidate nodes in the world adjacency matrices
    #         src_is_cand = candidates[src_idx]
    #         dst_is_cand = candidates[dst_idx]
    #
    #         # figure out which candidates have enough edges between them in world
    #         enough_edges = None
    #         for tmplt_adj, world_adj in iter_adj_pairs(tmplt, world):
    #             tmplt_adj_val = tmplt_adj[src_idx, dst_idx]
    #
    #             # if there are no edges in this channel of the template, skip it
    #             if tmplt_adj_val == 0:
    #                 continue
    #
    #             # sub adjacency matrix corresponding to edges from the source
    #             # candidates to the destination candidates
    #             world_sub_adj = world_adj[:, dst_is_cand][src_is_cand, :]
    #
    #             partial_enough_edges = world_sub_adj >= tmplt_adj_val
    #             if enough_edges is None:
    #                 enough_edges = partial_enough_edges
    #             else:
    #                 enough_edges = enough_edges.multiply(partial_enough_edges)
    #
    #         # # i,j element is 1 if cands i and j have enough edges between them
    #         # enough_edges = reduce(mul, enough_edges_list, 1)
    #
    #         # srcs with at least one reasonable dst
    #         src_matches = enough_edges.getnnz(axis=1) > 0
    #         candidates[src_idx][src_is_cand] = src_matches
    #         if not any(src_matches):
    #             candidates[:,:] = False
    #             break
    #
    #         if src_idx != dst_idx:
    #             # dsts with at least one reasonable src
    #             dst_matches = enough_edges.getnnz(axis=0) > 0
    #             candidates[dst_idx][dst_is_cand] = dst_matches
    #             if not any(dst_matches):
    #                 candidates[:,:] = False
    #                 break
    #
    #     return tmplt, world, candidates

    def __str__(self):
        """Summarize the state of the matching problem.

        Includes information about candidates for each template node.
        """
        # Append info strings to this list throughout the function.
        info_strs = []

        info_strs.append("There are {} template nodes and {} world nodes."
                         .format(self.tmplt.n_nodes, self.world.n_nodes))

        # Wouldn't want to recompute this too often.
        candidates = self.candidates()

        # Number of candidates for each template node.
        cand_counts = candidates.sum(axis=1)

        # TODO: if multiple nodes have the same candidates, condense them.

        # Iterate over template nodes in decreasing order of candidates.
        for idx in np.flip(np.argsort(cand_counts)):
            node = self.tmplt.nodes[idx]
            cands = sorted(self.world.nodes[candidates[idx]])
            n_cands = len(cands)

            if n_cands == 1:
                continue

            if n_cands > self._candidate_print_limit:
                cands = cands[:self._candidate_print_limit] + ["..."]

            # TODO: abstract out the getting and setting before and after
            info_strs.append("{} has {} candidates: {}"
                             .format(node, n_cands, ", ".join(cands)))

        # Nodes that have only one candidate
        identified = list(self.tmplt.nodes[cand_counts == 1])
        n_found = len(identified)

        # If there are any nodes that have only one candidate, that is
        # important information and should be recorded.
        if n_found:
            info_strs.append("{} template nodes have 1 candidate: {}"
                             .format(n_found, ", ".join(identified)))

        # This message is useful for debugging datasets for which you have
        # a ground truth signal.
        if self._ground_truth_provided:
            # Assuming ground truth nodes have same names, get the nodes for
            # which ground truth identity is not a candidate
            missing_ground_truth = [
                node for idx, node in enumerate(self.tmplt.nodes)
                if node not in self.world.nodes[candidates[idx]]
            ]
            n_missing = len(missing_ground_truth)

            info_strs.append("{} nodes are missing ground truth candidate: {}"
                             .format(n_missing, missing_ground_truth))

        return "\n".join(info_strs)
