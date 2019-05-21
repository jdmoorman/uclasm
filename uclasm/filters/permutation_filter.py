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

def gac_filter(tmplt, world, candidates, *,
                       changed_cands=None, verbose=False):
    """
    If k nodes in the template have the same k candidates, then those candidates
    are eliminated as candidates for all other template nodes

    Extension for the full GAC(AllDiff) constraint: considers also unions of sets
    """
    # The i'th element of this array is the number of cands for tmplt.nodes[i]
    cand_counts = np.sum(candidates, axis=1)

    # Nodes that have already been found to be permutable
    matched = np.zeros(tmplt.n_nodes, dtype=np.bool)

    # Run normal permutation filter while keeping track of matches
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
            matched[matches] = True

        # If more template nodes share the candidates than there are candidates
        # to be shared there can't be any isomorphisms.
        if match_count > cand_count:
            candidates[:,:] = False
            return tmplt, world, candidates

    cands_to_tmplt_set = {}
    n_matched = np.sum(matched)

    # Iterate over all remaining unions
    for node_idx, cand_count in enumerate(cand_counts):
        if cand_count >= tmplt.n_nodes - n_matched or matched[node_idx]:
            continue
        cand_set = frozenset(np.argwhere(candidates[node_idx]).flat)
        if cand_set in cands_to_tmplt_set:
            for other_cand_set in cands_to_tmplt_set:
                if cand_set.issubset(other_cand_set):
                    cands_to_tmplt_set[other_cand_set].add(node_idx)
            continue

        key_sets = [x for x in cands_to_tmplt_set.keys()]
        for key_set in key_sets:
            if cand_set.isdisjoint(key_set): # Skip disjoint sets
                continue
            if len(cand_set | key_set) >= tmplt.n_nodes:
                continue
            cands_to_tmplt_set[cand_set | key_set] = cands_to_tmplt_set[key_set] | {node_idx}
        cands_to_tmplt_set[cand_set] = {node_idx}
    for cand_set, tmplt_set in cands_to_tmplt_set.items():
        if len(cand_set) == len(tmplt_set):
            is_cand_row = np.zeros(candidates.shape[1], dtype=np.bool)
            is_cand_row[list(cand_set)] = True
            non_tmplt_array = np.ones(candidates.shape[0], dtype=np.bool)
            non_tmplt_array[list(tmplt_set)] = False
            candidates[np.ix_(non_tmplt_array, is_cand_row)] = False
    return tmplt, world, candidates
