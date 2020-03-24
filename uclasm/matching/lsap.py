"""Functions for solving variants of the linear sum assignment problem."""
import numpy as np
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment

from uclasm.utils import one_hot


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
    sub_row_ind, sub_col_ind = linear_sum_assignment(sub_costs)

    # Total cost is that of the submatrix lsap plus the cost of the constraint.
    return sub_costs[sub_row_ind, sub_col_ind].sum() + costs[i, j]


def constrained_lsap_costs(costs):
    """Solve a constrained linear sum assignment problem for each entry.

    TODO: More thorough testing of this function.

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
    # Since there are at least as many columns as rows, lsap_row_idxs should
    # be identical to np.arange(n_rows). We depend on this.
    lsap_row_idxs, lsap_col_idxs = linear_sum_assignment(costs)

    # Column vector of costs of each assignment in the lsap solution.
    lsap_costs = costs[lsap_row_idxs, lsap_col_idxs]
    lsap_total_cost = lsap_costs.sum()

    # When we add the constraint assigning row i to column j, lsap_col_idxs[i]
    # is freed up. If lsap_col_idxs[i] cannot improve on the cost of one of the
    # other row assignments, it does not need to be reassigned to another row.
    # If additionally column j is not in lsap_col_idxs, it is not taken away
    # from any of the other row assignments. In this situation, the resulting
    # total assignment costs are:
    total_costs = lsap_total_cost - lsap_costs[:, None] + costs

    # Handle cases where the column j is freed up and can be reused.
    for i, j in enumerate(lsap_col_idxs):
        # Can column j be reassigned to improve the assignment when row i is
        # constrained to another column?
        freed_col_costs = costs[:, j]
        if np.any(freed_col_costs < lsap_costs):
            print(i)
            # TODO: function for the subroutine below.
            # Solve the lsap with row i omitted. For the majority of
            # constraints on row i's assignment, this will not conflict with
            # the constraint. When it does conflict, we fix the issue later.
            sub_ind = ~one_hot(i, n_rows)
            sub_costs = costs[sub_ind, :][:, lsap_col_idxs]
            sub_row_ind, sub_col_ind = linear_sum_assignment(sub_costs)
            sub_total_cost = sub_costs[sub_row_ind, sub_col_ind].sum()

            # This calculation could be wrong for columns in lsap_col_idxs.
            # We handle such situations later.
            total_costs[i, :] = costs[i, :] + sub_total_cost

    sorted_costs_ind = np.argsort(costs, axis=1)
    """
    for lsap_i, lsap_j in lsap_row_idxs, lsap_col_idxs
      true_total_cost[lsap_i, lsap_j] == lsap_total_cost

    best_unused_j
    if costs[i, best_unused_j] - costs[i, best_used_j] < \
    true_total_costs[best_used_i, best_used_j] - true_total_costs[best_used_i, ?]
    then
    true_total_costs[i, best_unused_j] < true_total_costs[i, best_used_j]?
    """

    # When a row has its column stolen by a constraint, these are the columns
    # that might come into play when we are forced to resolve the assignment.
    potential_cols = list(lsap_col_idxs)
    for i, ordered_cols in enumerate(sorted_costs_ind):
        for j in ordered_cols:
            if j not in lsap_col_idxs:
                potential_cols.append(j)
                break

    # Handle constraints where the row i has its column stolen.
    for i, j in enumerate(tqdm(lsap_col_idxs, ascii=True, leave=True)):
        # Can row i be reassigned to a new column easily when its first choice
        # j is constrained to another row? i.e. is its next best option among
        # the columns already in use?

        # What is the next best assignment for this row?
        best_j = sorted_costs_ind[i, 0]
        second_best_j = sorted_costs_ind[i, 1]
        next_best_j = best_j if best_j != j else second_best_j

        # Is the next best option available?
        if next_best_j in lsap_col_idxs:
            for k in tqdm(lsap_row_idxs, ascii=True, leave=False):
                if k == i:
                    continue
                # TODO: Can we avoid solving the constrained lsap again?
                sub_costs = costs[:, potential_cols]
                sub_j = potential_cols.index(j)
                total_costs[k, j] = constrained_lsap_cost(k, sub_j, sub_costs)
        else:
            total_costs[:, j] += costs[i, next_best_j] - costs[i, j]

    np.set_printoptions(linewidth=1000, precision=2, floatmode='fixed')
    print(costs)
    print(lsap_col_idxs)
    for i, j in zip(lsap_row_idxs, lsap_col_idxs):
        if total_costs[i, j] != lsap_total_cost:
            print(i, j, total_costs[i, j], lsap_total_cost)

    # # Any constraints which do not conflict with the unconstrained lsap
    # # solution share the total cost of the unconstrained lsap solution.
    # total_costs[lsap_row_idxs, lsap_col_idxs] = lsap_total_cost

    return total_costs
