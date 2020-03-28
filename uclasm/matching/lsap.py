"""Functions for solving variants of the linear sum assignment problem."""
import numpy as np
from tqdm import tqdm
import functools

from scipy.optimize import linear_sum_assignment as lap

from uclasm.utils import one_hot

# ASCII matches the progress bar of dask.
tqdm = functools.partial(tqdm, ascii=True)


def constrained_lsap_cost(i, j, costs):
    """Compute the total cost of a constrained linear sum assignment problem.

    Parameters
    ----------
    i : int
        Row index corresponding to the constraint.
    j : int
        Column index corresponding to the constraint.
    costs : 2darray
        A matrix of costs.

    Returns
    -------
    float
        The total cost of the linear sum assignment problem solution under the
        constraint that row i is assigned to column j.
    """
    n_rows, n_cols = costs.shape

    # Cost matrix omitting the row and column corresponding to the constraint.
    sub_costs = costs[~one_hot(i, n_rows), :][:, ~one_hot(j, n_cols)]

    # Lsap solution for the submatrix.
    sub_row_ind, sub_col_ind = lap(sub_costs)

    # Total cost is that of the submatrix lsap plus the cost of the constraint.
    return sub_costs[sub_row_ind, sub_col_ind].sum() + costs[i, j]


def constrained_lsap_costs(costs):
    """Solve a constrained linear sum assignment problem for each entry.

    TODO: More thorough testing of this function.

    The output of this function is equivalent to, but significantly more
    efficient than,

    >>> def constrained_lsap_costs(costs):
    ...     total_costs = np.empty_like(costs)
    ...     for i, j in np.ndindex(*costs.shape):
    ...         total_costs[i, j] = constrained_lsap_cost(i, j, costs)
    ...     return total_costs

    Parameters
    ----------
    costs : 2darray
        A matrix of costs.

    Returns
    -------
    2darray
        A matrix of total constrained lsap costs. The i, j entry of the matrix
        corresponds to the total lsap cost under the constraint that row i is
        assigned to column j.
    """
    n_rows, n_cols = costs.shape
    if n_rows > n_cols:
        return constrained_lsap_costs(costs.T).T

    # Find the best lsap assignment from rows to columns without constrains.
    # Since there are at least as many columns as rows, row_idxs should
    # be identical to np.arange(n_rows). We depend on this.
    row_idxs, lsap_col_idxs = lap(costs)

    # Column vector of costs of each assignment in the lsap solution.
    lsap_costs = costs[row_idxs, lsap_col_idxs]
    lsap_total_cost = lsap_costs.sum()

    # Each row is column indexes ordered by their costs in that row.
    sorted_costs_ind = np.argsort(costs, axis=1)

    # When a row has its column stolen by a constraint, these are the columns
    # that might come into play when we are forced to resolve the assignment.
    used = np.isin(sorted_costs_ind[:, :n_rows+1], lsap_col_idxs)
    first_unused = sorted_costs_ind[row_idxs, np.argmax(~used, axis=1)]
    potential_cols = np.union1d(lsap_col_idxs, first_unused)

    # When we add the constraint assigning row i to column j, lsap_col_idxs[i]
    # is freed up. If lsap_col_idxs[i] cannot improve on the cost of one of the
    # other row assignments, it does not need to be reassigned to another row.
    # If additionally column j is not in lsap_col_idxs, it is not taken away
    # from any of the other row assignments. In this situation, the resulting
    # total assignment costs are:
    total_costs = lsap_total_cost - lsap_costs[:, None] + costs

    for i, freed_j in enumerate(tqdm(lsap_col_idxs)):
        # For each row, which column is it currently assigned to? Modify this
        # as we go to enforce various constraints. Set the row i entry to -1
        # to indicate that we are enforcing constraints on row i at the moment.
        col_idxs = lsap_col_idxs.copy()
        col_idxs[i] = -1  # Row i is having a constraint applied.

        # When row i is constrained to another column, can column j be
        # reassigned to improve the assignment cost of one of the other rows?
        freed_col_costs = costs[:, freed_j]
        if np.any(freed_col_costs < lsap_costs):
            # Solve the lsap with row i omitted. For the majority of
            # constraints on row i's assignment, this will not conflict with
            # the constraint. When it does conflict, we fix the issue later.
            sub_ind = ~one_hot(i, n_rows)
            sub_costs = costs[sub_ind, :][:, lsap_col_idxs]
            sub_row_ind, sub_col_ind = lap(sub_costs)
            sub_total_cost = sub_costs[sub_row_ind, sub_col_ind].sum()
            col_idxs[sub_ind] = lsap_col_idxs[sub_col_ind]

            # This calculation will end up being wrong for the columns in
            # lsap_col_idxs[sub_col_ind]. This is because the constraint in
            # row i in these columns will conflict with the sub assignment.
            # These miscalculations are corrected later.
            total_costs[i, :] = costs[i, :] + sub_total_cost

        # col_idxs now contains the optimal assignment columns ignoring row i.
        col_idxs[i] = np.setdiff1d(lsap_col_idxs, col_idxs)[0]
        total_costs[i, col_idxs[i]] = costs[row_idxs, col_idxs].sum()
        col_idxs[i] = -1

        for other_i, stolen_j in enumerate(tqdm(col_idxs, leave=False)):
            if other_i == i:
                continue

            # Row i steals column stolen_j from other_i because of constraint.
            col_idxs[i] = stolen_j
            col_idxs[other_i] = -1

            # Row other_i must find a new column. What is its next best option?
            best_j = sorted_costs_ind[other_i, 0]
            second_best_j = sorted_costs_ind[other_i, 1]
            next_best_j = best_j if (best_j != stolen_j) else second_best_j

            # Is the next best option available? If so, use it. Otherwise,
            # solve the constrained lsap from scratch.
            if next_best_j not in col_idxs:
                col_idxs[other_i] = next_best_j
                total_costs[i, stolen_j] = costs[row_idxs, col_idxs].sum()
            else:
                sub_costs = costs[:, potential_cols]
                sub_j = np.argwhere(potential_cols == stolen_j)[0]
                total_cost = constrained_lsap_cost(i, sub_j, sub_costs)
                total_costs[i, stolen_j] = total_cost

            # Give other_i its column back in preparation for the next round.
            col_idxs[other_i] = stolen_j
            col_idxs[i] = -1

    # For those constraints which are compatible with the unconstrained lsap:
    total_costs[row_idxs, lsap_col_idxs] = lsap_total_cost

    return total_costs
