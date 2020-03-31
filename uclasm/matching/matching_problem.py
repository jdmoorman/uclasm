"""This module provides a class for representing subgraph matching problems."""
import numpy as np
from loguru import logger

from .lsap import constrained_lsap_costs


class MatchingProblem:
    """A class representing any subgraph matching problem, noisy or otherwise.

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
    fixed_costs : 2darray
        Cost of assigning a template node to a world node, ignoring structure.
        One row for each template node, one column for each world node.
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
        World graph to be searched.
    fixed_costs : 2darray
        Cost of assigning a template node to a world node, ignoring structure.
        One row for each template node, one column for each world node.
    structural_costs : 2darray
        Each entry of this matrix denotes the minimum local cost of matching
        the template node corresponding to the row to the world node
        corresponding to the column.
    cost_threshold : int, optional
        A subgraph whose cost againt the template exceeds this threshold will
        not be considered a match. It can also be used to eliminate candidates
        from the world graph. A cost of 0 corresponds to an exact match for the
        template, whereas a cost of 1 means that the match may be missing a
        single edge which is present in the template but not in the world.
    """

    def __init__(self,
                 tmplt, world,
                 fixed_costs=None,
                 cost_threshold=0,
                 ground_truth_provided=False,
                 candidate_print_limit=10):
        self.tmplt = tmplt
        tmplt_channels = set(self.tmplt.channels)
        world_channels = set(world.channels)
        if tmplt_channels != world_channels:
            logger.warning("World channels {} do not appear in template.",
                           world_channels - tmplt_channels)
        self.world = world.channel_subgraph(self.tmplt.channels)

        shape = (tmplt.n_nodes, world.n_nodes)
        self.structural_costs = np.zeros(shape)
        if fixed_costs is None:
            self.fixed_costs = np.zeros(shape)
            self._total_costs = np.zeros(shape)
        else:
            self.fixed_costs = fixed_costs

        self._total_costs = self._compute_total_costs()
        self._structural_cost_sum = 0  # self.costs.sum()
        self.cost_threshold = cost_threshold
        self._ground_truth_provided = ground_truth_provided
        self._candidate_print_limit = candidate_print_limit

    def _have_costs_changed(self):
        """Check if the structural costs have changed since last call.

        Returns
        -------
        bool
            True if any of self.structural_costs have changed since last time
            this function was called. False otherwise.
        """
        old_structural_cost_sum = self._structural_cost_sum
        self._structural_cost_sum = self.structural_costs.sum()
        return old_structural_cost_sum != self._structural_cost_sum

    def _compute_total_costs(self):
        """Compute total costs from structural and fixed costs.

        Returns
        -------
        2darray
            [self.tmplt.n_nodes, self.world.n_nodes] array of total costs.
            Each entry constrains the template node corresponding to the row
            to be assigned to the world node corresponding to the column. The
            value of the entry is the minimum assignment cost under the
            corresponding constraint.
        """
        costs = self.structural_costs / 2 + self.fixed_costs
        return constrained_lsap_costs(costs)

    @property
    def total_costs(self):
        """2darray: A matrix of minimum costs under assignment constraints.

        Each entry of this 2darray is a lower bound on the total assignment
        cost that will be incurred by a subgraph match in which the template
        node corresponding to the row is assigned to the world node
        corresponding to the column.
        """
        if self._have_costs_changed():
            self._total_costs = self._compute_total_costs()

        return self._total_costs

    def candidates(self):
        """Get the matrix of compatibility between template and world nodes.

        World node j is considered to be a candidate for a template node i if
        there exists an assignment from template nodes to world nodes in which
        i is assigned to j whose cost does not exceed the desired threshold.

        This could be a property, but it is not particularly cheap to compute.

        Returns
        -------
        2darray
            A boolean matrix where each entry indicates whether the world node
            corresponding to the column is a candidate for the template node
            corresponding to the row.
        """
        return self.total_costs <= self.cost_threshold

    def update_costs(self, new_structural_costs, indexer=None):
        """Update the structural costs with the larger of the old and the new.

        Each entry of self.structural_costs is monotonically increasing.

        Parameters
        ----------
        new_structural_costs : ndarray
            Costs to update with. Any current structural_costs that are larger
            than thecorresponding new structural_costs are kept.
        indexer : ndarray
            The elements of self.structural_costs that are to be updated.
        """
        if indexer is None:
            self.structural_costs = np.maximum(self.structural_costs,
                                               new_structural_costs)
        else:
            self.structural_costs[indexer] = \
                np.maximum(self.structural_costs[indexer],
                           new_structural_costs)

    def __str__(self):
        """Summarize the state of the matching problem.

        Returns
        -------
        str
            Information includes number of candidates for each template node,
            number of template nodes which have exactly one candidate,
            and size of the template and world graphs.
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
