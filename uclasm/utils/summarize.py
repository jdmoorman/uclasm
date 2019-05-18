import numpy as np

def summarize(tmplt, world, candidates, alert_missing=True):
    cand_counts = candidates.sum(axis=1)

    # Nodes that have only one candidate
    identified = tmplt.nodes[cand_counts==1]
    n_found = len(identified)

    # Assuming ground truth nodes have same names, get the nodes for which
    # ground truth identity is not a candidate
    missing_ground_truth = [node for idx, node in enumerate(tmplt.nodes)
                            if node not in world.nodes[candidates[idx]]]
    n_missing = len(missing_ground_truth)

    # Use number of candidates to decide the order to print the summaries
    def key_func(node, cand_counts=cand_counts):
        return (-cand_counts[tmplt.node_idxs[node]], -tmplt.node_idxs[node])

    # TODO: if multiple nodes have the same candidates, condense them
    for node in sorted(tmplt.nodes, key=key_func):
        cands = world.nodes[candidates[tmplt.node_idxs[node]]]
        n_cands = len(cands)

        if n_cands == 1:
            continue

        # TODO: abstract out the getting and setting before and after
        print_opts = np.get_printoptions()
        np.set_printoptions(threshold=10, edgeitems=6)
        print(node, "has", n_cands, "candidates:", cands)
        np.set_printoptions(**print_opts)

    if n_found:
        print(n_found, "template nodes have 1 candidate:", identified)

    # This message is useful for debugging datasets for which you have
    # a ground truth signal
    if n_missing > 0 and alert_missing:
        print(n_missing, "nodes are missing ground truth candidate:",
              missing_ground_truth)
