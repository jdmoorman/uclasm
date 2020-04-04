"""Miscellaneous functions and helpers for the uclasm package."""
import numpy as np


def one_hot(idx, length):
    """Return a 1darray of zeros with a single one in the idx'th entry."""
    one_hot = np.zeros(length, dtype=np.bool)
    one_hot[idx] = True
    return one_hot


def index_map(args):
    """Return a dict mapping elements to their indices.

    Parameters
    ----------
    args : Iterable[str]
        Strings to be mapped to their indices.
    """
    return {elm: idx for idx, elm in enumerate(args)}


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


def apply_index_map_to_cols(df, cols, values):
    """Replace df[cols] with their indexes as taken from names.

    Parameters
    ----------
    df : DataFrame
        To be modified inplace.
    cols : Iterable[str]
        Columns of df to operate on.
    values : Iterable[str]
        Values expected to be present in df[cols] to be replaced with their
        corresponding indexes.
    """
    val_to_idx = index_map(values)
    df[cols] = df[cols].applymap(val_to_idx.get)
