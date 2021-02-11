""" 
Code to partition vertices into structural equivalent classes given a structural
equivalence partition per channel.
by TimNg
"""

from .equivalence_data_structure import Equivalence
from .partition_equivalence_vertices import partition_vertices


def multichannel_equivalence_relation(vertex_a, vertex_b, equiv_partition_dict):
    """ 
    Comparison function to run Equivalence.partition on equiv_partition_dict 
    is a dictionary of the form {'ch' -> Equivalence class}
    - two vertices are structurally equivalent in a multichannel graph iff 
    they are structurally equivalent in each channel 
    - we are given the structural partition in the equiv_partition_dict.
    we just need to check if 2 vertices are in the same equiv class 
    for every channel."""
    for ch, equiv_classes in equiv_partition_dict.items():
        if not equiv_classes.in_same_class(vertex_a, vertex_b):
            return False
    return True


def combine_channel_equivalence(equiv_partition_dict):
    """ 
    Args:
        equiv_partition_dict: dictionary of 'ch' : [{equiv_classes}]
        Output: Equivalence class object that contains the proper partition
        Use Equivalence.classes() to obtain a list 
            of sets of vertices in the same equiv classes for all channel
    """
    one_partition = next(iter(equiv_partition_dict.values()))
    result = Equivalence(list(one_partition.parent_map.keys()))
    result.partition(multichannel_equivalence_relation, equiv_partition_dict)
    return result


def partition_multichannel(ch_to_adj, vertices=None) -> Equivalence:
    """
    Args:
        ch_to_adj: Dictionary of 'ch': sparse adj_matrix
    Returns: Equivalence object that contains the proper partition
    """
    equiv_partition_dictionary = {}  # dictionary to hold partition per channel
    for ch, sparse_mat in ch_to_adj.items():
        adj_matrix = sparse_mat.toarray() # array is faster than sparse mat
        # compute the partition for that channel
        equiv_partition_dictionary[ch] = partition_vertices(adj_matrix, vertices) 
    return combine_channel_equivalence(equiv_partition_dictionary)
