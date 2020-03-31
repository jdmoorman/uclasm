"""
Implementation of functions to partition vertices according to permutation equivalency
for the DARPA Multichannel Directed Multigraph Subgraph Matching Project
by TimNg
"""

from .equivalence_data_structure import Equivalence
import numpy as np
from queue import Queue

def array_equal(mat1, mat2):
    """
    Check if two sparse matrices are equal.
    """
    return (mat1 != mat2).nnz==0

def permutation_relation(u, v, adj_matrix_csr,
        adj_matrix_csc, row_nnz, col_nnz):
    """
    We compare two vertices: a and b to see if they are equivalence
    under permutation. This can be done by checking if certain pieces
    of their values in the adj_matrix are the same (see notes)

    Args:
        u (int): The index of the first vertex
        v (int): The index of the second vertex
        adj_mat_csr (sparse_matrix): A csr representation of the adjacency
            matrix
        adj_mat_csc (sparse_matrix): A csc representation of the adjacency
            matrix
        row_nnz (array): An array of the number nonzero count of each row
        col_nnz (array): An array of the number nonzero count of each col
    """
    # handle the trivial case
    if u == v:
        return True

    u, v = (u ,v) if u < v else (v, u)

    if (col_nnz[u] == 0 and col_nnz[v] == 0 and 
            row_nnz[u] == 0 and row_nnz[v] == 0):
        continue

    n = adj_mat_csr.shape[0]
    
    if adj_mat_csr[u,u] != adj_mat_csr[v,v]:
        return False
    if adj_mat_csr[u,v] != adj_mat_csr[v,u]:
        return False

    # we do this check to avoid checking subarrays if we don't have to
    if row_nnz[u] != 0:
        # check the rows (3 pieces), maybe there are better ways to do this
        if not (array_equal(adj_mat_csr[u,0:u],   adj_mat_csr[v,0:u]) and
                array_equal(adj_mat_csr[u,u+1:v], adj_mat_csr[v,u+1:v]) and
                array_equal(adj_mat_csr[u,v+1:n], adj_mat_csr[v,v+1:n])):
            return False
    # check the columns
    if col_nnz[u] != 0:
        if not (array_equal(adj_mat_csc[0:u,u],   adj_mat_csc[0:u,v]) and
                array_equal(adj_mat_csc[u+1:v,u], adj_mat_csc[u+1:v,v]) and
                array_equal(adj_mat_csc[v+1:n,u], adj_mat_csc[v+1:n,v])):
            return False
    # we have passed all the test if we reach here... return True now
    return True


def partition_vertices(adj_matrix: '2d numpy array') -> [{'equiv-vertices'}]:
    """
    Given an adj_matrix (directed multigraph) we partition the vertices 
    (by indices) in terms of the permutation relation.
    """
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "Must be square matrix!"
    # initialize an equivalence relation fill with nodes
    nodes = range(0, adj_matrix.shape[0])
    adj_matrix_csc = adj_matrix.tocsc()
    row_nnz = adj_matrix.getnnz(1)
    col_nnz = adj_matrix_csc.getnnz(0)
    vertices_partition = Equivalence(nodes)
    # partition into equivalence classes
    vertices_partition.partition(permutation_relation, adj_matrix, 
                                 adj_matrix_csc, row_nnz, col_nnz)
    return vertices_partition


def equivalent(u, v, adj_mat_csrs, adj_mat_cscs, row_nnzs, col_nnzs):
    """
    Check if vertex u is equivalent to vertex v. This does this by checking
    if the respective entries of the adjacency matrices match. This does this
    with respect to all channels.

    Args:
        u (int): The index of the first vertex
        v (int): The index of the second vertex
        adj_mat_csrs (dict): A dictionary from channels to csr representations
            of the matrices
        adj_mat_cscs (dict): A dictionary from channels to csc representations
            of the matrices
        row_nnzs (dict): A dictionary from channels to an array of the
            number nonzero count of each row
        col_nnzs (dict): A dictionary from channels to an array of the
            number nonzero count of each column
    
    Returns:
        bool: A flag indicating if u is equivalent to v
    """
    if u == v:
        return True

    # We do preliminary checks to rule out as many equivalences as possible
    # because the later checks are expensive
    # We first check that the degrees match
    for channel in adj_mat_csrs:
        row_nnz = row_nnzs[channel]
        col_nnz = col_nnzs[channel]
        if row_nnz[u] != row_nnz[v] or col_nnz[u] != col_nnz[v]:
            return False

    for channel in adj_mat_csrs:
        if not permutation_relation(u, v, adj_mat_csrs[channel], 
                adj_mat_cscs[channel], row_nnzs[channel], col_nnzs[channel]):
            return False
    return True


def partition_neighbors(neighbors, adj_mat_csrs, adj_mat_cscs, 
                        row_nnzs, col_nnzs):
    """
    Construct the equivalence classes of the passed in vertices.

    Args:
        neighbors (list): A list of the indices of the neighbors
        adj_mat_csrs (dict): A dictionary from channels to csr representations
            of the matrices
        adj_mat_cscs (dict): A dictionary from channels to csc representations
            of the matrices
        row_nnzs (dict): A dictionary from channels to an array of the
            number nonzero count of each row
        col_nnzs (dict): A dictionary from channels to an array of the
            number nonzero count of each column
    """
    eq_classes = [[neighbors[0]]]
    for neighbor in neighbors[1:]:
        # See if a neighbor is in one of the already computed eq classes
        for eq_class in eq_classes:
            if equivalent(neighbor, eq_class[0], adj_mat_csrs, adj_mat_cscs,
                          row_nnzs, col_nnzs):
                eq_class.append(neighbor)
                break
        else:
            # Otherwise it is in its own class
            eq_classes.append([neighbor])

    return eq_classes


def bfs_partition_graph(ch_to_adj):
    """
    This performs a breadth first search approach to computing the equivalence
    classes. This function works by performing a breadth first search of the
    vertices; At each step it forms equivalence classes of the neighbors 
    (No vertex that is not a neighbor can be equivalent to any of the neighbors
     so there is no need to check outside the neighbors)
    In the BFS, we also only need to visit one representative of each 
    equivalence class.

    It then needs to do a last check to find which class the inital vertex is
    in.

    Args:
        ch_to_adj (dict): A dictionary from channels to adjacency matrices
            of the graph
    Returns:
        Equivalence: The equivalence structure of the graph
    """
    # We create a queue to perform a Breadth First Search
    queue = Queue()
    visited = np.zeros(graph.n_nodes, dtype=np.bool)
    visited[0] = True
    all_eq_classes = []
    # We start at vertex 0
    queue.put(0)

    # We create these beforehand so as to only compute them once
    adj_mat_csrs = graph.ch_to_adj
    adj_mat_cscs = {ch: adj_mat.tocsc() for ch, adj_mat in adj_mat_csrs.items()}
    row_nnzs = {ch: adj_mat.getnnz(1) for ch, adj_mat in adj_mat_csrs.items()}
    col_nnzs = {ch: adj_mat.getnnz(0) for ch, adj_mat in adj_mat_cscs.items()}

    while True:
        # If out of vertices to visit, exit loop
        if queue.empty():
            break
        curr_vertex = queue.get()

        neighbors = graph.sym_composite_adj[curr_vertex].nonzero()[1]
        neighbors = neighbors[~visited[neighbors]]
        visited[neighbors] = True
        if len(neighbors) == 0:
            continue
        eq_classes = partition_neighbors(neighbors, adj_mat_csrs, adj_mat_cscs,
                                         row_nnzs, col_nnzs)
        all_eq_classes.extend(eq_classes)
        # For each equivalence class, we put a representative
        for eq_class in eq_classes:
            queue.put(eq_class[0])
    
    # We still need to find which equivalence class 0 is in
    for eq_class in all_eq_classes:
        if equivalent(0, eq_class[0], adj_mat_csrs, adj_mat_cscs, 
                      row_nnzs, col_nnzs):
            eq_class.append(0)
            break
    else:
        # 0 is in its own class
        all_eq_classes.append([0])

    # Construct the Equivalence object.
    equivalence = Equivalence(list(range(graph.n_nodes)))
    for eq_class in all_eq_classes:
        x = eq_class[0]
        for y in eq_class[1:]:
            equivalence.merge_classes_of(x, y)

    return equivalence
