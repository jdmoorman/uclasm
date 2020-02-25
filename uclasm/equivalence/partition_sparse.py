''' Implementation of functions to partition vertices according to permutation equivalency
    for the DARPA Multichannel Directed Multigraph Subgraph Matching Project
    (this works for directed multigraph. To be generalized to multichannel case)
    by TimNg
    Last Update: 7/8/19 (Basic Functionalities)'''

from .equivalence_data_structure import Equivalence
import numpy as np
from queue import Queue

def array_equal(mat1, mat2):
    ''' special array_equal for sparse matrix'''
    return (mat1 != mat2).nnz==0

def permutation_relation(vertex_a, vertex_b, adj_matrix: 'sparse matrix',
        adj_matrix_csc: 'sparse matrix', row_nnz, col_nnz):
    ''' 
    We compare two vertices: a and b to see if they are equivalence
    under permutation. This can be done by checking if certain pieces
    of their values in the adj_matrix are the same (see notes)
    '''
    n = adj_matrix.shape[0] # get the shape
    # handle the trivial case
    if vertex_a == vertex_b:
        return True
    # in the paper x < y so we use similar notation
    x,y = (vertex_a,vertex_b) if vertex_a < vertex_b else (vertex_b, vertex_a)
    if col_nnz[x] == 0 and col_nnz[y] == 0 and row_nnz[x] == 0 and row_nnz[x] == 0:
        return True
    # several conditions to check
    if row_nnz[x] != row_nnz[y] or col_nnz[x] != col_nnz[y]:
        return False
    if adj_matrix[x,x] != adj_matrix[y,y]:
        return False
    if adj_matrix[x,y] != adj_matrix[y,x]:
        return False
    # check the rows (3 pieces) -- maybe there are better ways to do this
    if not (array_equal(adj_matrix[x,0:x],   adj_matrix[y,0:x]) and \
            array_equal(adj_matrix[x,x+1:y], adj_matrix[y,x+1:y]) and\
            array_equal(adj_matrix[x,y+1:n], adj_matrix[y,y+1:n])):
        return False
    # check the columns
    if not (array_equal(adj_matrix_csc[0:x,x],   adj_matrix_csc[0:x,y]) and \
            array_equal(adj_matrix_csc[x+1:y,x], adj_matrix_csc[x+1:y,y]) and \
            array_equal(adj_matrix_csc[y+1:n,x], adj_matrix_csc[y+1:n,y])):
        return False
    # we have passed all the test if we reach here... return True now
    return True

def partition_vertices(adj_matrix: '2d numpy array') \
    -> [{'equiv-vertices'}]:
    ''' Given an adj_matrix (directed multigraph)
    we partition the vertices (by indices) in terms of the permutation relation'''
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "partition_vertices: Must be square matrix!"
    # initialize an equivalence relation fill with nodes
    nodes = range(0, adj_matrix.shape[0])
    adj_matrix_csc = adj_matrix.tocsc()
    row_nnz = adj_matrix.getnnz(1)
    col_nnz = adj_matrix_csc.getnnz(0)
    vertices_partition = Equivalence(nodes)
    # partition into equivalence classes
    vertices_partition.partition(permutation_relation, adj_matrix, adj_matrix_csc,
                                 row_nnz, col_nnz)
    return vertices_partition


def equivalent(u, v, adj_mat_csrs, adj_mat_cscs, row_nnzs, col_nnzs):
    """
    Check if vertex u is equivalent to vertex v
    """
    if u == v:
        return True

    # Swap them if v > u
    u, v = (u, v) if u < v else (v, u)

    # We do preliminary checks to rule out as many equivalences as possible
    # because the later checks are expensive
    # We check that the degrees match
    for channel in adj_mat_csrs:
        row_nnz = row_nnzs[channel]
        col_nnz = col_nnzs[channel]
        if row_nnz[u] != row_nnz[v] or col_nnz[u] != col_nnz[v]:
            return False

    for channel in adj_mat_csrs:
        row_nnz = row_nnzs[channel]
        col_nnz = col_nnzs[channel]
        if col_nnz[u] == 0 and col_nnz[v] == 0 and row_nnz[u] == 0 and row_nnz[v] == 0:
            continue

        adj_mat_csr = adj_mat_csrs[channel]
        adj_mat_csc = adj_mat_cscs[channel]
        n = adj_mat_csr.shape[0]
        
        if adj_mat_csr[u,u] != adj_mat_csr[v,v]:
            return False
        if adj_mat_csr[u,v] != adj_mat_csr[v,u]:
            return False

        # we do this check to avoid checking subarrays if we don't have to
        if row_nnz[u] != 0:
            # check the rows (3 pieces) -- maybe there are better ways to do this
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
    return True


def partition_neighbors(neighbors, adj_mat_csrs, adj_mat_cscs, row_nnzs, col_nnzs):
    eq_classes = [[neighbors[0]]]
    for neighbor in neighbors[1:]:
        for eq_class in eq_classes:
            if equivalent(neighbor, eq_class[0], adj_mat_csrs, adj_mat_cscs,
                          row_nnzs, col_nnzs):
                eq_class.append(neighbor)
                break
        else:
            eq_classes.append([neighbor])

    return eq_classes


def bfs_partition_graph(graph):
    queue = Queue()
    visited = np.zeros(graph.n_nodes, dtype=np.bool)
    visited[0] = True
    all_eq_classes = []
    # We start at vertex 0
    queue.put(0)

    adj_mat_csrs = graph.ch_to_adj
    adj_mat_cscs = {ch: adj_mat.tocsc() for ch, adj_mat in adj_mat_csrs.items()}
    row_nnzs = {ch: adj_mat.getnnz(1) for ch, adj_mat in adj_mat_csrs.items()}
    col_nnzs = {ch: adj_mat.getnnz(0) for ch, adj_mat in adj_mat_cscs.items()}

    count = 0
    while True:
        if queue.empty():
            break
        curr_vertex = queue.get()

        neighbors = graph.sym_composite_adj[curr_vertex].nonzero()[1]
        neighbors = neighbors[~visited[neighbors]]
        count += len(neighbors)
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

    equivalence = Equivalence(list(range(graph.n_nodes)))
    for eq_class in all_eq_classes:
        x = eq_class[0]
        for y in eq_class[1:]:
            equivalence.merge_classes_of(x, y)

    return equivalence
