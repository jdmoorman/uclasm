"""Miscellaneous functions for the uclasm package."""
import numpy as np


def one_hot(idx, length):
    """Return a 1darray of zeros with a single one in the idx'th entry."""
    one_hot = np.zeros(length, dtype=np.bool)
    one_hot[idx] = True
    return one_hot


def index_map(arg_list):
    """Return a dict mapping elements of the list to their indices."""
    return {elm: idx for idx, elm in enumerate(arg_list)}


# TODO: change the name of this function
def invert(dict_of_sets):
    """TODO: Docstring."""
    new_dict = {}
    for k, v in dict_of_sets.items():
        for x in v:
            new_dict[x] = new_dict.get(x, set()) | set([k])
    return new_dict


def values_map_to_same_key(dict_of_sets):
    """TODO: Docstring."""
    matches = {}

    # get the sets of candidates
    for key, val_set in dict_of_sets.items():
        frozen_val_set = frozenset(val_set)
        matches[frozen_val_set] = matches.get(frozen_val_set, set()) | {key}

    return matches
