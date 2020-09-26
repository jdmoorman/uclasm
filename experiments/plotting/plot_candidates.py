import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import sys
sys.path.append('../')
import uclasm

def plot_candidates(tmplt, candidates, results_dir, fig_name, label_type="cand_count"):
    """
    Plot the tmplt with nodes labeling 0-N, ordered by # of candidates left
    """
    nxgraph = nx.from_scipy_sparse_matrix(tmplt.composite_adj,
                                          create_using=nx.DiGraph())

    n_components = nx.number_weakly_connected_components(nxgraph)
    tmplt_pos = {}
    subgraphs = list(nx.weakly_connected_component_subgraphs(nxgraph))

    for subfig_i, subgraph in enumerate(subgraphs):
        df = pd.DataFrame(index=nxgraph.nodes(), columns=nxgraph.nodes())
        for row, data in nx.shortest_path_length(subgraph.to_undirected()):
            for col, dist in data.items():
                df.loc[row,col] = dist
        pos = nx.kamada_kawai_layout(subgraph, dist=df.to_dict())

        subfig_center = np.array([2.5*subfig_i, 0])
        pos = {key: subfig_center + val for key, val in pos.items()}

        tmplt_pos.update(pos)

    cand_counts = np.sum(candidates, axis=1)
    if label_type == "cand_count":
        labels = {i: cand_counts[i] for i in range(tmplt.n_nodes)}
    elif label_type == "cand_order":
        x = sorted(range(tmplt.n_nodes), key=lambda idx: cand_counts[idx])
        labels = {i : x.index(i) for i in range(tmplt.n_nodes)}
    else:
        labels = {i: node for i, node in enumerate(tmplt.nodes)}

    nx.draw(nxgraph,
            node_size=1000,
            edge_color="black",
            node_color="#26FEFD",
            labels=labels,
            pos=tmplt_pos)

    plt.savefig('{}/{}_{}.png'.format(results_dir, fig_name, label_type))
