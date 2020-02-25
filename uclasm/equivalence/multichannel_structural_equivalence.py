""" Code to partition vertices into structural equivalent classes given a structural
    equivalence partition per channel.
    by TimNg
    Last Update: 7/10/19"""

from .equivalence_data_structure import Equivalence
from .partition_equivalence_vertices import partition_vertices # this is for 1 channel


def multichannel_equivalence_relation(vertex_a, vertex_b, equiv_partition_dict) -> bool:
    """ comparison function to run Equivalence.partition on
        equiv_partition_dict is a dictionary of the form:
            'ch' : Equivalence class
        - two vertices are structurally equivalent in a multichannel graph iff 
            they are structurally equivalent in each channel 
        --> we are given the structural partition in the equiv_partition_dict.
            we just need to check if 2 vertices are in the same equiv class for every channel"""
    for ch, equiv_classes in equiv_partition_dict.items():
        if not equiv_classes.in_same_class(vertex_a, vertex_b):  # O(1)
            return False
    return True


def combine_channel_equivalence(equiv_partition_dict):
    """ Input: dictionary of 'ch' : [{equiv_classes}]
        Output: Equivalence class object that contains the proper partition
        Use Equivalence.classes() to obtain a list 
            of sets of vertices in the same equiv classes for all channel
    """
    one_partition = next(iter(equiv_partition_dict.values()))
    result = Equivalence(list(one_partition.parent_map.keys()))
    result.partition(multichannel_equivalence_relation, equiv_partition_dict)
    return result


# this version is for the format used in the darpa project --> can be made to general matrix
def partition_multichannel(ch_to_adj, vertices=None) -> Equivalence:
    """Input: Dictionary of 'ch': sparse adj_matrix
        Output: Equivalence class object that contains the proper partition ->
        use .classes() to obtain a list of sets of vertices in the same equivalence classes (structural equivalence)"""
    equiv_partition_dictionary = {}  # dictionary to hold partition per channel
    for ch, sparse_mat in ch_to_adj.items():
        adj_matrix = sparse_mat.toarray()  # array is a lot faster than sparse mat
        equiv_partition_dictionary[ch] = partition_vertices(adj_matrix, vertices)  # compute the partition for that channel
    return combine_channel_equivalence(equiv_partition_dictionary)
