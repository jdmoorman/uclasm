"""Helpers for the MatchingProblem class."""
import numpy as np
from loguru import logger


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