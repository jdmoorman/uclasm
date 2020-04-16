"""Helpers for the MatchingProblem class."""
import numpy as np
from loguru import logger
import numba


def inspect_channels(tmplt, world):
    """Check if the channels of the template and world graph are compatible.

    In particular, the channels of the template should be a subset of those in
    the world. Otherwise, there cannot possibly be a match.

    TODO: Should we pad with empty channels?

    Parameters
    ----------
    tmplt : Graph
        Template graph to be matched.
    world : Graph
        World graph to be searched.
    """
    tmplt_channels = set(tmplt.channels)
    world_channels = set(world.channels)
    if tmplt_channels != world_channels:
        logger.warning("World channels {} do not appear in template.",
                       world_channels - tmplt_channels)

    if not tmplt_channels.issubset(world_channels):
        logger.error("Template channels {} do not appear in world.",
                     tmplt_channels - world_channels)


class MonotoneArray(np.ndarray):
    """An ndarray whose entries cannot decrease.

    Example
    -------
    >>> A = np.zeros(3).view(MonotoneArray)
    >>> A[0:2] = [-1, 1]
    >>> A
    MonotoneArray([0.,  1.,  0.])

    """

    def __setitem__(self, key, value):
        """Ensure values cannot decrease."""
        value = np.maximum(self[key], value)
        super().__setitem__(key, value)

class GlobalCostsArray(MonotoneArray):
    """An array storing the global costs that automatically updates a boolean
    array of candidates based on a global cost threshold.

    Attributes
    ----------
    global_cost_threshold : int
        The global cost threshold to compare to.
    candidates : ndarray(bool)
        A boolean array of candidates with the same shape as the costs array.

    """
    def __new__(cls, input_array, global_cost_threshold=0, candidates=None):
        """Initialize the global costs array.
        Parameters
        ----------
        input_array : ndarray
            The original array of costs
        global_cost_threshold : int
            The global cost threshold to compare to.
        candidates : ndarray(bool)
            A boolean array of candidates with the same shape as the costs array.
        """
        obj = np.asarray(input_array).view(cls)
        obj.global_cost_threshold = global_cost_threshold
        if candidates is not None:
            obj.candidates = candidates
            if candidates.shape != obj.shape:
                raise Exception("Shape of provided candidates array does not match.")
        else:
            obj.candidates = np.ones(obj.shape, dtype=np.bool)
        return obj

    def set_global_cost_threshold(self, new_global_cost_threshold):
        """Sets a new global cost threshold and updates the candidates."""
        self.global_cost_threshold = new_global_cost_threshold
        self.candidates[self > self.global_cost_threshold] = False

    def __array_finalize__(self, obj):
        """NumPy method called when new instance created."""
        if obj is None:
            return
        self.global_cost_threshold = getattr(obj, 'global_cost_threshold', 0)
        self.candidates = getattr(obj, 'candidates', None)
        if self.candidates is None:
            self.candidates = np.ones(self.shape, dtype=np.bool)

    def __getitem__(self, key):
        """Handle candidates when indexing with slices."""
        test = super().__getitem__(key)
        if isinstance(test, GlobalCostsArray):
            if test.shape != test.candidates.shape:
                test.candidates = test.candidates[key]
        return test

    def __setitem__(self, key, value):
        """Update candidates whenever a new cost is set."""
        super().__setitem__(key, value)
        self.candidates[key] = np.logical_and(self.candidates[key],
                                              self[key]<=self.global_cost_threshold)

@numba.njit(parallel=True)
def feature_disagreements(tmplt_features, world_features):
    """Compute the amount by which the template's features exceed the world's.

    The feature disagreement is computed separately in each channel and summed.

    Parameters
    ----------
    tmplt_features : 2darray
        [n_tmplt_nodes, n_features] array of features for each template node.
    world_features : 2darray
        [n_world_nodes, n_features] array of features for each world node.

    Returns
    -------
    2darray
        [n_tmplt_nodes, n_world_nodes] array of amounts by which each template
        node's features exceeded each world node's features summed across all
        of the features.

    Notes
    -----
    Do not be tempted to use a tensorized implementation of this function.
    It will cause the memory usage of this function to explode leading to
    slower performance.
    """
    n_tmplt_nodes = tmplt_features.shape[0]
    n_world_nodes = world_features.shape[0]

    # Preallocate memory
    disagreements = np.empty((n_tmplt_nodes, n_world_nodes))

    # For each template and world node, compute the disagreement between their
    # features. Currently, the amount by which the template features exceed
    # the world features.
    for tidx in numba.prange(n_tmplt_nodes):
        tnode_features = tmplt_features[tidx, :]

        for widx in numba.prange(n_world_nodes):
            wnode_features = world_features[widx, :]

            # TODO: Insert generic loss function between features here
            disagreement = np.maximum(tnode_features - wnode_features, 0).sum()
            disagreements[tidx, widx] = disagreement

    return disagreements
