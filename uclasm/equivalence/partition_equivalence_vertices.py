""" Implementation of functions to partition vertices according to permutation equivalency
    for the DARPA Multichannel Directed Multigraph Subgraph Matching Project
    (this works for directed multigraph. To be generalized to multichannel case)
    by TimNg
    Last Update: 8/7/19 (Extended for vertices)"""

from .equivalence_data_structure import Equivalence
import numpy as np


def permutation_relation(vertex_a, vertex_b, adj_matrix):
    """ nodes and adj_matrix are global variable
    We compare two vertices: a and b to see if they are equivalence
    under permutation. This can be done by checking if certain pieces
    of their values in the adj_matrix are the same (see notes)"""
    n = adj_matrix.shape[0] 
    assert vertex_a in range(0, n), \
        "permutation_relation: invalid index for vertex " + str(vertex_a)
    assert vertex_b in range(0, n), \
        "permutation_relation: invalid index for vertex " + str(vertex_b)
    # handle the trivial case
    if vertex_a == vertex_b:
        return True
    # in the paper x < y so we use similar notation
    x,y = (vertex_a,vertex_b) if vertex_a < vertex_b else (vertex_b, vertex_a)
    # several conditions to check
    if adj_matrix[x,x] != adj_matrix[y, y]:
        return False
    if adj_matrix[x,y] != adj_matrix[y, x]:
        return False
    # check the rows (3 pieces) -- maybe there are better ways to do this
    if not (np.array_equal(adj_matrix[x, 0:x],   adj_matrix[y, 0:x]) and
            np.array_equal(adj_matrix[x, x+1:y], adj_matrix[y, x+1:y]) and
            np.array_equal(adj_matrix[x, y+1:n], adj_matrix[y, y+1:n])):
        return False
    # check the columns
    if not (np.array_equal(adj_matrix[0:x, x],   adj_matrix[0:x, y]) and
            np.array_equal(adj_matrix[x+1:y, x], adj_matrix[x+1:y, y]) and
            np.array_equal(adj_matrix[y+1:n, x], adj_matrix[y+1:n, y])):
        return False
    # we have passed all the test if we reach here... return True now
    return True


def partition_vertices(adj_matrix: '2d numpy array', vertices=None) -> Equivalence:
    """ Given a set of vertices and an adj_matrix (directed multigraph)
    we partition them in terms of the permutation relation.
    If vertices is none, we partition all nodes of that graph"""
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "partition_vertices: Input matrix must be square!"
    if vertices is None:
        # initialize an equivalence relation fill with nodes
        vertices_partition = Equivalence(range(0, adj_matrix.shape[0]))
    else:
        vertices_partition = Equivalence(vertices)
    # partition into equivalence classes
    vertices_partition.partition(permutation_relation, adj_matrix)
    return vertices_partition
