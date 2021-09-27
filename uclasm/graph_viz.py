import numpy as np
import matplotlib
import graphviz as gv


def get_gv_graph(graph, eq_classes=None, with_node_labels=False, with_edge_labels=False,
                 node_key='rdf:type', edge_key='rdf:type'):
    cmap1 = plt.get_cmap('Set1')
    cmap3 = plt.get_cmap('Set3')
    if eq_classes:
        nontrivial_classes = [eq_class for eq_class in eq_classes if len(eq_class) > 1]

    gv_graph = gv.Digraph()
    for i in range(graph.n_nodes):
        node_name = graph.nodelist[node_key][i]
        if eq_classes:
            eq_class = [eq_class for eq_class in eq_classes if i in eq_class][0]
            if eq_class in nontrivial_classes:
                idx = nontrivial_classes.index(eq_class)
                color = matplotlib.colors.rgb2hex(cmap3.colors[idx % len(cmap3.colors)])
            else:
                idx = 8,
                color = matplotlib.colors.rgb2hex(cmap1.colors[idx])
        else:
            color = matplotlib.colors.rgb2hex(cmap1.colors[8])
        
        if with_node_labels:
            gv_graph.node(str(i), label=node_name, color=color, style='filled')
        else:
            gv_graph.node(str(i), color=color, style='filled')

    for v1, v2 in zip(graph.edgelist['node1'], graph.edgelist['node2']):
        v1_index, v2_index = graph.node_idxs[v1], graph.node_idxs[v2]
        if with_edge_labels:
            edge = graph.edgelist[(graph.edgelist['node1'] == v1)
                                  & (graph.edgelist['node2'] == v2)][edge_key]
            gv_graph.edge(str(v1_index), str(v2_index), label=str(list(edge)[0]))
        else:
            gv_graph.edge(str(v1_index), str(v2_index))

    gv_graph.engine = 'fdp'
    return gv_graph

