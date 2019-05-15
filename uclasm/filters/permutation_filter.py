import numpy as np

def permutation_filter(tmplt, world, candidates, *,
                       changed_cands=None, verbose=False):
    """
    If k nodes in the template have the same k candidates, then those candidates
    are eliminated as candidates for all other template nodes
    """
    # The i'th element of this array is the number of cands for tmplt.nodes[i]
    cand_counts = np.sum(candidates, axis=1)

    for node_idx, cand_count in sorted(enumerate(cand_counts), key=lambda x: -x[1]):
        # Any set of candidates larger than the template can be skipped
        if cand_count >= tmplt.n_nodes:
            continue

        is_cand_row = candidates[node_idx]

        # matches correspond to other template nodes who share the same cands
        # or whose cands are a subset of node_idx's cands
        matches = np.sum(candidates[:, is_cand_row], axis=1) == cand_counts

        # How many template nodes share these cands?
        match_count = np.sum(matches)

        # If the number of template nodes sharing the candidates is the same
        # as the number of candidates
        if match_count == cand_count:
            # Eliminate the candidates for all other template nodes
            for non_match_idx in np.argwhere(~matches).flat:
                candidates[non_match_idx, is_cand_row] = False

        # If more template nodes share the candidates than there are candidates
        # to be shared there can't be any isomorphisms.
        if match_count > cand_count:
            candidates[:,:] = False
            break

    return tmplt, world, candidates
