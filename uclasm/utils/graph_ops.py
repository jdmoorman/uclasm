from .misc import one_hot
import numpy as np

# TODO: return the indices in the order they were found. most to least neighbors
# TODO: make a function for getting a node cover, use that function here.
def get_unspec_cover(tmplt):
    """
    get a reasonably small set of template nodes which, if removed, would cause
    all of the remaining template nodes with multiple candidates to become
    disconnected
    """
    # Ones correspond to template nodes with at least one candidate
    unspec = tmplt.get_cand_counts() > 1

    # Initially there are no nodes in the cover. We add them one by one below.
    uncovered = unspec.copy()

    # Until the cover disconnects the unspec nodes, add a node to the cover
    while tmplt.is_nbr[uncovered, :][:, uncovered].count_nonzero():
        # Add the unspec node with the most neighbors to the cover
        nbr_counts = np.sum(tmplt.is_nbr[uncovered, :][:, uncovered], axis=0)

        # TODO: make this line less ugly
        uncovered[uncovered] = ~one_hot(np.argmax(nbr_counts),
                                        np.sum(uncovered))

    return np.argwhere(unspec ^ uncovered).flat
