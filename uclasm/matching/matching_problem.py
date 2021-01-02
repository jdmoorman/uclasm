"""This module provides a class for representing subgraph matching problems."""
import numpy as np
import os

from .matching_utils import inspect_channels, MonotoneArray, \
    feature_disagreements
from .global_cost_bound import *

class MatchingProblem:
    """A class representing any subgraph matching problem, noisy or otherwise.

    TODO: describe the class in more detail.
    TODO: optionally accept ground truth map argument.
    TODO: Is it okay to describe the tmplt and world attributes using the same
    descriptions as were used for the corresponding parameters?
    TODO: Introduce local_cost threshold.

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
    fixed_costs : 2darray, optional
        Cost of assigning a template node to a world node, ignoring structure.
        One row for each template node, one column for each world node.
    local_costs : 2darray, optional
        Initial local costs.
    global_costs : 2darray, optional
        Initial global costs.
    node_attr_fn : function
        Function for comparing node attributes. Should take two pd.Series of
        node attributes and return the cost associated with the difference
        between them.
    edge_attr_fn : function
        Function for comparing edge attributes. Should take two pd.Series of
        edge attributes and return the cost associated with the difference
        between them.
    missing_edge_cost_fn : function
        Function for computing the cost of a missing template edge. Should take
        a pd.Series of node attributes and return the cost associated with
        removing that edge.
    local_cost_threshold : int, optional
        A template node cannot be assigned to a world node if it will result
        in more than this number of its edges missing in an eventual match.
    global_cost_threshold : int, optional
        A subgraph whose cost againt the template exceeds this threshold will
        not be considered a match. It can also be used to eliminate candidates
        from the world graph. A cost of 0 corresponds to an exact match for the
        template, whereas a cost of 1 means that the match may be missing a
        single edge which is present in the template but not in the world.
    ground_truth_provided : bool, optional
        A flag indicating whether a signal has been injected into the world
        graph with node identifiers that match those in the template.
    candidate_print_limit : int, optional
        When summarizing the candidates of each template node, limit the list
        of candidates to this many.
    use_monotone : bool, optional
        Whether to use monotone arrays for the cost. Defaults to true.

    Attributes
    ----------
    tmplt : Graph
        Template graph to be matched.
    world : Graph
        World graph to be searched.
    shape : (int, int)
        Size of the matching problem: Number of template nodes and world nodes.
    local_cost_threshold : int, optional
        A template node cannot be assigned to a world node if it will result
        in more than this number of its edges missing.
    global_cost_threshold : int, optional
        A subgraph whose cost againt the template exceeds this threshold will
        not be considered a match. It can also be used to eliminate candidates
        from the world graph. A cost of 0 corresponds to an exact match for the
        template, whereas a cost of 1 means that the match may be missing a
        single edge which is present in the template but not in the world.
    """

    def __init__(self,
                 tmplt, world,
                 fixed_costs=None,
                 local_costs=None,
                 global_costs=None,
                 node_attr_fn=None,
                 edge_attr_fn=None,
                 missing_edge_cost_fn=None,
                 local_cost_threshold=0,
                 global_cost_threshold=0,
                 strict_threshold=False,
                 ground_truth_provided=False,
                 candidate_print_limit=10,
                 cache_path=None,
                 edgewise_costs_cache=None,
                 use_monotone=True,
                 match_fixed_costs=False):

        # Various important matrices will have this shape.
        self.shape = (tmplt.n_nodes, world.n_nodes)

        if fixed_costs is None:
            fixed_costs = np.zeros(self.shape)

        if local_costs is None:
            local_costs = np.zeros(self.shape)

        if global_costs is None:
            global_costs = np.zeros(self.shape)

        # Make sure graphs have the same channels in the same order.
        if tmplt.channels != world.channels:
            inspect_channels(tmplt, world)
            world = world.channel_subgraph(tmplt.channels)

        # Account for self edges in fixed costs.
        if tmplt.adjs is not None and tmplt.has_loops:
            fixed_costs += feature_disagreements(
                tmplt.self_edges,
                world.self_edges
            )
            tmplt = tmplt.loopless_subgraph()
            world = world.loopless_subgraph()

        self.use_monotone = use_monotone
        self.match_fixed_costs = match_fixed_costs

        if use_monotone:
            self._fixed_costs = fixed_costs.view(MonotoneArray)
            self._local_costs = local_costs.view(MonotoneArray)
            # self._global_costs = global_costs.view(MonotoneArray)
        else:
            self._fixed_costs = fixed_costs
            self._local_costs = local_costs
            # self._global_costs = global_costs
        self._global_costs = global_costs.view(MonotoneArray)

        # No longer care about self-edges because they are fixed costs.
        self.tmplt = tmplt
        self.world = world

        # Cache of edge-to-edge costs for the edgewise local cost bound
        self._edgewise_costs_cache = edgewise_costs_cache
        self.cache_path = cache_path
        if self.cache_path is not None and self._edgewise_costs_cache is None:
            try:
                self._edgewise_costs_cache = np.load(os.path.join(self.cache_path, "edgewise_costs_cache.npy"))
                n_tmplt_edges = len(self.tmplt.edgelist.index)
                n_world_edges = len(self.world.edgelist.index)
                if self._edgewise_costs_cache.shape != (n_tmplt_edges, n_world_edges):
                    tmplt_edge_to_attr_idx = np.load(os.path.join(self.cache_path, "tmplt_edge_to_attr_idx.npy"))
                    world_edge_to_attr_idx = np.load(os.path.join(self.cache_path, "world_edge_to_attr_idx.npy"))
                    print('Edge to attr maps loaded from cache')
                    
                    self.tmplt_edge_to_attr_idx = tmplt_edge_to_attr_idx
                    self.world_edge_to_attr_idx = world_edge_to_attr_idx

                    if len(self.tmplt_edge_to_attr_idx) != n_tmplt_edges or len(self.world_edge_to_attr_idx) != n_world_edges:
                        raise Exception("Edgewise costs cache not properly computed!")
                print("Edge-to-edge costs loaded from cache")
            except IOError as e:
                print("No edgewise cost cache found.")

        self.local_cost_threshold = local_cost_threshold
        self.global_cost_threshold = global_cost_threshold
        self.strict_threshold = strict_threshold

        self._ground_truth_provided = ground_truth_provided
        self._candidate_print_limit = candidate_print_limit

        self._num_valid_candidates = self.tmplt.n_nodes * self.world.n_nodes

        self.node_attr_fn = node_attr_fn
        self.edge_attr_fn = edge_attr_fn
        self.missing_edge_cost_fn = missing_edge_cost_fn

        self.matching = tuple()
        self.assigned_tmplt_idxs = set()

    def copy(self, copy_graphs=True):
        """Returns a copy of the MatchingProblem."""
        if copy_graphs:
            tmplt = self.tmplt.copy()
            world = self.world.copy()
        else:
            tmplt = self.tmplt
            world = self.world
        smp_copy = MatchingProblem(tmplt, world,
            fixed_costs=self._fixed_costs.copy() if self.match_fixed_costs else self._fixed_costs,
            local_costs=self._local_costs.copy(),
            global_costs=self._global_costs.copy(),
            node_attr_fn=self.node_attr_fn,
            edge_attr_fn=self.edge_attr_fn,
            missing_edge_cost_fn=self.missing_edge_cost_fn,
            local_cost_threshold=self.local_cost_threshold,
            global_cost_threshold=self.global_cost_threshold,
            strict_threshold=self.strict_threshold,
            ground_truth_provided=self._ground_truth_provided,
            candidate_print_limit=self._candidate_print_limit,
            cache_path=self.cache_path,
            edgewise_costs_cache=self._edgewise_costs_cache,
            use_monotone=self.use_monotone,
            match_fixed_costs=self.match_fixed_costs)
        if hasattr(self, "template_importance"):
            smp_copy.template_importance = self.template_importance
        if hasattr(self, "tmplt_edge_to_attr_idx"):
            smp_copy.tmplt_edge_to_attr_idx = self.tmplt_edge_to_attr_idx.copy()
        if hasattr(self, "world_edge_to_attr_idx"):
            smp_copy.world_edge_to_attr_idx = self.world_edge_to_attr_idx.copy()
        if hasattr(self.tmplt, "time_constraints"):
            smp_copy.tmplt.time_constraints = self.tmplt.time_constraints
        if hasattr(self.tmplt, "geo_constraints"):
            smp_copy.tmplt.geo_constraints = self.tmplt.geo_constraints
        return smp_copy

    def set_costs(self, fixed_costs=None, local_costs=None, global_costs=None):
        """Set the cost arrays by force. Override monotonicity.

        Parameters
        ----------
        fixed_costs : 2darray, optional
        local_costs : 2darray, optional
        global_costs : 2darray, optional

        """
        if self.use_monotone:
            if fixed_costs is not None:
                self._fixed_costs = fixed_costs.view(MonotoneArray)

            if local_costs is not None:
                self._local_costs = local_costs.view(MonotoneArray)

            if global_costs is not None:
                self._global_costs = global_costs.view(MonotoneArray)
        else:
            if fixed_costs is not None:
                self._fixed_costs = fixed_costs

            if local_costs is not None:
                self._local_costs = local_costs

            if global_costs is not None:
                self._global_costs = global_costs

    @property
    def fixed_costs(self):
        """2darray: Fixed costs such as node attribute mismatches.

        Cost of assigning a template node to a world node, ignoring structure.
        One row for each template node, one column for each world node.
        TODO: Better docstrings.
        """
        return self._fixed_costs

    @fixed_costs.setter
    def fixed_costs(self, value):
        self._fixed_costs[:] = value

    @property
    def local_costs(self):
        """2darray: Local costs such as missing edges around each node.

        Each entry of this matrix denotes a bound on the local cost of matching
        the template node corresponding to the row to the world node
        corresponding to the column.
        TODO: Better docstrings.
        """
        return self._local_costs

    @local_costs.setter
    def local_costs(self, value):
        if self._local_costs is not None:
            self._local_costs[:] = value
        else:
            self._local_costs = value

    @property
    def global_costs(self):
        """2darray: Costs of full graph match.

        Each entry of this matrix bounds the global cost of matching
        the template node corresponding to the row to the world node
        corresponding to the column.
        TODO: Better docstrings.
        """
        return self._global_costs

    @global_costs.setter
    def global_costs(self, value):
        self._global_costs[:] = value

    def candidates(self, tmplt_idx=None):
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
        if self.strict_threshold:
            # return np.logical_and(self.global_costs < self.global_cost_threshold,
            #                       ~np.isclose(self.global_costs, self.global_cost_threshold))
            if tmplt_idx is not None:
                return self.global_costs[tmplt_idx] < (self.global_cost_threshold - 1e-8)
            return self.global_costs < (self.global_cost_threshold - 1e-8)
        # return np.logical_or(self.global_costs <= self.global_cost_threshold,
        #                      np.isclose(self.global_costs, self.global_cost_threshold))
        if tmplt_idx is not None:
            return self.global_costs[tmplt_idx] <= (self.global_cost_threshold + 1e-8)
        return self.global_costs <= (self.global_cost_threshold + 1e-8)

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

            if n_cands <= 1:
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

        # Nodes that have no candidates
        unidentified = list(self.tmplt.nodes[cand_counts == 0])
        n_unidentified = len(unidentified)
        if n_unidentified:
            info_strs.append("{} template nodes have 0 candidates: {}"
                             .format(n_unidentified, ", ".join(unidentified)))

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

    def reduce_world(self):
        """Reduce the size of the world graph.

        Check whether there are any world nodes that are not candidates to
        any tmplt nodes. If so, remove them from the world graph and update
        the matching problem.

        Returns
        -------
        np.ndarray(bool)
            A boolean array of values corresponding to which nodes were kept.
            True where nodes were kept and false where they were removed.
        """
        # Note: need to update the global_costs before reduce_world to reflect
        # changes in the candidates

        is_cand = self.candidates().any(axis=0)

        # If some world node does not serve as candidates to any tmplt node
        if ~is_cand.all():
            # Update matching
            new_matching = []
            for tmplt_idx, world_idx in self.matching:
                if is_cand[world_idx]:
                    new_world_idx = int(np.sum(is_cand[:world_idx]))
                    new_matching.append((tmplt_idx, new_world_idx))
            self.matching = tuple(new_matching)

            self.world, edge_is_cand = self.world.node_subgraph(is_cand, get_edge_is_cand=True)
            self.shape = (self.tmplt.n_nodes, self.world.n_nodes)

            # Update parameters based on new world
            self.set_costs(local_costs=self.local_costs[:, is_cand])
            self.set_costs(fixed_costs=self.fixed_costs[:, is_cand])
            self.set_costs(global_costs=self.global_costs[:, is_cand])
            from_local_bounds(self)

            if edge_is_cand is not None and self._edgewise_costs_cache is not None:
                if hasattr(self, 'world_edge_to_attr_idx'):
                    self.world_edge_to_attr_idx = self.world_edge_to_attr_idx[edge_is_cand]
                else:
                    self._edgewise_costs_cache = self._edgewise_costs_cache[:, edge_is_cand]
        return is_cand

    def have_candidates_changed(self):
        """Check whether candidates have changed.

        Returns
        -------
        bool
            True if any of the candidates have been eliminated. False otherwise.
        """
        # TODO: this function needs to be updated
        num_valid_candidates = self._num_valid_candidates
        self._num_valid_candidates = np.count_nonzero(self.candidates())
        return num_valid_candidates != self._num_valid_candidates

    def add_match(self, tmplt_idx, world_idx):
        """Enforce that the template node with the given index must match the
        corresponding world node with the given index.
        Parameters
        ----------
        tmplt_idx : int
            The index of the template node to be matched.
        world_idx : int
            The index of the world node to be matched.
        """
        new_matching = [match for match in self.matching]
        new_matching.append((tmplt_idx, world_idx))
        self.enforce_matching(tuple(new_matching))

    def enforce_matching(self, matching):
        """Enforce the given matching tuple by setting fixed costs in off-match
        rows and columns to float("inf")
        Parameters
        ----------
        matching : iterable
            Iterable of 2-tuples indicating pairs of template-world indexes
        """
        self.matching = matching
        if self.match_fixed_costs:
            mask = self.get_non_matching_mask()
            self.fixed_costs[mask] = float("inf")
        self.assigned_tmplt_idxs = {tmplt_idx for tmplt_idx, cand_idx in self.matching}

    def get_non_matching_mask(self):
        """Gets a boolean mask for the costs array corresponding to all entries
        that would violate the matching."""
        mask = np.zeros(self.fixed_costs.shape, dtype=np.bool)
        if len(self.matching) > 0:
            mask[[pair[0] for pair in self.matching],:] = True
            mask[:,[pair[1] for pair in self.matching]] = True
            mask[tuple(np.array(self.matching).T)] = False
        return mask

    def prevent_match(self, tmplt_idx, world_idx):
        """Prevent matching the template node with the given index to the world
        node with the given index.
        Parameters
        ----------
        tmplt_idx : int
            The index of the template node not to be matched.
        world_idx : int
            The index of the world node not to be matched.
        """
        self.fixed_costs[tmplt_idx, world_idx] = float("inf")
