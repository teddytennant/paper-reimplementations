"""Exact linear assignment, written out rather than imported.

This is the decoder in hLLM. Equation (1) of the paper is

    pi* = argmax_{pi in Pi_N} sum_i M_{i, pi(i)},

the maximum weight bipartite matching between the N items and the K rank
positions of the score matrix M. Section 3.2 calls this the linear assignment
problem and points at Kuhn (1955) for the O(N^3) Hungarian algorithm and at
Jonker and Volgenant (1987) for the shortest augmenting path variant the paper
actually times (Table 5: 0.008 ms at N = 50).

What follows is that shortest augmenting path form, in numpy, so it can be read
next to the paper. It keeps dual potentials u over rows and v over columns and
grows one alternating tree per row, which is the Kuhn-Munkres invariant stated
in the dual: u_i + v_j <= C_ij everywhere, with equality on matched pairs.
Cost of a row scan is O(K) and each of the N rows augments once, so the whole
solve is O(N^2 K).

The solver minimizes a cost matrix. Equation (1) maximizes, so the decoder
passes it -M.
"""

from __future__ import annotations

import numpy as np

__all__ = ["solve_assignment", "assign_positions", "decode_ranking", "assignment_score"]


def _lap_min(cost: np.ndarray) -> np.ndarray:
    """Minimum cost assignment of every row of ``cost`` to a distinct column.

    Requires ``n <= m``. Returns ``col`` with ``col[i]`` the column matched to
    row i.

    The loop is the Jonker-Volgenant shortest augmenting path: for the newly
    added row we run a Dijkstra over columns in the reduced cost
    ``C_ij - u_i - v_j``, which is non-negative by the dual invariant, then
    shift the potentials by the shortest distance ``delta`` so that the whole
    scanned tree stays tight and the found path can be flipped.
    """
    n, m = cost.shape
    if n > m:
        raise ValueError(f"_lap_min needs n <= m, got {cost.shape}")

    inf = np.inf
    # 1-based rows and columns, index 0 is the sentinel "unmatched" slot that
    # the augmenting path starts from.
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    parent = np.zeros(m + 1, dtype=np.int64)  # parent[j] = row matched to column j
    way = np.zeros(m + 1, dtype=np.int64)  # predecessor column on the shortest path

    for i in range(1, n + 1):
        parent[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = parent[j0]
            # Relax every unvisited column through the row currently sitting on
            # the frontier column j0.
            reduced = cost[i0 - 1] - u[i0] - v[1:]
            better = ~used[1:] & (reduced < minv[1:])
            minv[1:] = np.where(better, reduced, minv[1:])
            way[1:] = np.where(better, j0, way[1:])

            candidates = np.where(used[1:], inf, minv[1:])
            j1 = int(np.argmin(candidates)) + 1
            delta = candidates[j1 - 1]
            if not np.isfinite(delta):
                raise ValueError("cost matrix must be finite")

            # Potential shift: scanned rows go up by delta, scanned columns go
            # down by delta, unscanned distances shrink by delta. Reduced costs
            # inside the tree are unchanged, so matched pairs stay tight.
            scanned = np.flatnonzero(used)
            u[parent[scanned]] += delta
            v[scanned] -= delta
            minv[~used] -= delta

            j0 = j1
            if parent[j0] == 0:
                break

        # Flip the path: every column on it takes the row of its predecessor,
        # and the free row i ends up matched.
        while j0:
            j1 = way[j0]
            parent[j0] = parent[j1]
            j0 = j1

    col = np.full(n, -1, dtype=np.int64)
    matched = np.flatnonzero(parent[1:] != 0) + 1
    col[parent[matched] - 1] = matched - 1
    return col


def solve_assignment(cost, maximize: bool = False):
    """Solve the linear assignment problem on an ``(n, m)`` cost matrix.

    Returns ``(row_ind, col_ind)``, both sorted by ``row_ind``, matching the
    convention of ``scipy.optimize.linear_sum_assignment``. The matching has
    ``min(n, m)`` pairs, is injective in both directions, and is optimal.

    ``maximize=True`` solves equation (1) directly instead of minimizing.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2-D, got shape {cost.shape}")
    if cost.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    if not np.all(np.isfinite(cost)):
        raise ValueError("cost matrix must be finite")

    work = -cost if maximize else cost
    n, m = work.shape
    if n <= m:
        col = _lap_min(work)
        return np.arange(n, dtype=np.int64), col
    row = _lap_min(np.ascontiguousarray(work.T))  # row[j] = row matched to column j
    order = np.argsort(row, kind="stable")
    return row[order], order.astype(np.int64)


def assign_positions(scores) -> np.ndarray:
    """Decode equation (1): the rank position each item is assigned.

    ``scores`` is the N x K item-position score matrix M. Returns an int array
    of length N holding the position of each item, or -1 for items left over
    when N > K. The result is a valid injective assignment by construction,
    which is the structural guarantee of Section 6.2; nothing is repaired
    afterwards.
    """
    scores = np.asarray(scores, dtype=np.float64)
    rows, cols = solve_assignment(scores, maximize=True)
    pos = np.full(scores.shape[0], -1, dtype=np.int64)
    pos[rows] = cols
    return pos


def decode_ranking(scores) -> np.ndarray:
    """Decode the emitted ordinals: item index at each rank position.

    Returns an int array of length ``min(N, K)`` holding the items in order of
    the position they were assigned. This is the ranked list at the right of
    Figure 1. With N >= K, the paper's case, entry j is the item at rank j.
    With fewer items than positions the assignment can skip positions, and the
    result is still the items in rank order.
    """
    scores = np.asarray(scores, dtype=np.float64)
    rows, cols = solve_assignment(scores, maximize=True)
    order = np.argsort(cols, kind="stable")
    return rows[order]


def assignment_score(scores, pos) -> float:
    """Objective value of equation (1) for a given assignment.

    ``pos`` is in the format of :func:`assign_positions`; entries of -1 are
    unmatched items and contribute nothing.
    """
    scores = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(pos)
    rows = np.flatnonzero(pos >= 0)
    return float(scores[rows, pos[rows]].sum())
