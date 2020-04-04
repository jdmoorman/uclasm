"""Provide a function for bounding node assignment costs with nodewise info."""

from ..matching_utils import feature_disagreements


def nodewise(smp):
    """Bound local assignment costs by comparing in and out degrees.

    TODO: Cite paper from REU.
    TODO: Take candidacy into account when computing features.
    TODO: Bring back reciprocated edges, carefully?

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute nodewise cost bounds.
    """
    smp.local_costs = feature_disagreements(
        smp.tmplt.in_out_degrees,
        smp.world.in_out_degrees
    )
