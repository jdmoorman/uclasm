import numpy as np

def one_hot(idx, length):
    one_hot = np.zeros(length, dtype=np.bool)
    one_hot[idx] = True
    return one_hot

def index_map(arg_list):
    """return a dict mapping elements of the list to their indices"""
    return {elm: idx for idx, elm in enumerate(arg_list)}

# TODO: change the name of this function
def invert(dict_of_sets):
    new_dict = {}
    for k,v in dict_of_sets.items():
        for x in v:
            new_dict[x] = new_dict.get(x, set()) | set([k])
    return new_dict

def values_map_to_same_key(dict_of_sets):
    matches = {}

    # get the sets of candidates
    for key, value_set in dict_of_sets.items():
        frozen_value_set = frozenset(value_set)
        matches[frozen_value_set] = matches.get(frozen_value_set, set()) | {key}

    return matches
