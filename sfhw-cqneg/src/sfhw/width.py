"""Fractional covers and signed fractional hypertree width.

rho* is the fractional edge cover number from Section 3.3. Bags are the
canonical variable-elimination bags of Section 3.1. sfhw is Definition 3.7:
one variable order must work for every subset of the negative edges.

The cover solver enumerates basic feasible solutions of the primal LP with
exact rational arithmetic. It is exact for every hypergraph it accepts
(|V| <= 6 and |E| <= 8 after discarding edges that miss the covered set).
Tests only need values in {1, 1.5, 2}. Orders are enumerated only for
|V| <= 5, matching the task bound.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction


class WidthLimitError(ValueError):
    """Raised when an instance is outside the exact solver's stated bounds."""


def _as_edge(edge):
    return frozenset(edge)


def _solve_unit_system(matrix):
    """Solve A w = 1 over Fractions. Return None if A is singular."""
    k = len(matrix)
    if k == 0:
        return []
    aug = [list(row) + [Fraction(1)] for row in matrix]
    for col in range(k):
        pivot = None
        for row in range(col, k):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        for j in range(col, k + 1):
            aug[col][j] /= scale
        for row in range(k):
            if row == col or aug[row][col] == 0:
                continue
            factor = aug[row][col]
            for j in range(col, k + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][k] for i in range(k)]


def rho_star(edges, vertices):
    """Fractional edge cover number rho*_H(X) from Section 3.3.

    edges may be any iterable of vertex sets. Only vertices in `vertices`
    are required to be covered, and an edge covers the vertices it contains.
    Returns a float. math.inf if some vertex lies on no edge.
    """
    target = frozenset(vertices)
    if not target:
        return 0.0
    if len(target) > 6:
        raise WidthLimitError("rho* solver is exact only for |V| <= 6")

    reduced = []
    seen = set()
    for edge in edges:
        piece = _as_edge(edge) & target
        if piece and piece not in seen:
            seen.add(piece)
            reduced.append(piece)
    if len(reduced) > 8:
        raise WidthLimitError("rho* solver is exact only for |E| <= 8")

    covered = set()
    for edge in reduced:
        covered |= edge
    if not target <= covered:
        return math.inf

    vertices_l = list(target)
    n = len(vertices_l)
    m = len(reduced)
    best = Fraction(n)

    for k in range(1, min(n, m) + 1):
        for basis_idx in itertools.combinations(range(m), k):
            basis = [reduced[i] for i in basis_idx]
            for tight in itertools.combinations(range(n), k):
                matrix = []
                for ti in tight:
                    matrix.append(
                        [1 if vertices_l[ti] in basis[j] else 0 for j in range(k)]
                    )
                weights = _solve_unit_system(matrix)
                if weights is None or any(w < 0 for w in weights):
                    continue
                if any(
                    sum(weights[j] for j in range(k) if vertex in basis[j]) < 1
                    for vertex in vertices_l
                ):
                    continue
                total = sum(weights, Fraction(0))
                if total < best:
                    best = total
    return float(best)


def canonical_bags(order, edges):
    """Bags chi_sigma(sigma_i) from the Section 3.1 elimination order.

    H_0 = (V, E). At step i, U_i is the union of edges containing sigma_i,
    chi(sigma_i) = U_i, those edges are deleted, and U_i without sigma_i is
    added when it is non-empty.
    """
    remaining = [set(edge) for edge in edges]
    bags = {}
    for variable in order:
        incident = [edge for edge in remaining if variable in edge]
        bag = set()
        for edge in incident:
            bag |= edge
        if not bag:
            bag = {variable}
        bags[variable] = frozenset(bag)
        remaining = [edge for edge in remaining if variable not in edge]
        leftover = bag - {variable}
        if leftover:
            remaining.append(leftover)
    return bags


def _width_for_order(variables, pos_edges, neg_edges, order):
    """max over S and i of rho*(chi_sigma^S(sigma_i))."""
    worst = 0.0
    neg_list = list(neg_edges)
    for r in range(len(neg_list) + 1):
        for subset in itertools.combinations(neg_list, r):
            edges = list(pos_edges) + list(subset)
            bags = canonical_bags(order, edges)
            for variable in variables:
                cover = rho_star(edges, bags[variable])
                if cover > worst:
                    worst = cover
    return worst


def sfhw_edges(variables, pos_edges, neg_edges):
    """Signed fractional hypertree width, Definition 3.7.

    sfhw = min_sigma max_S max_i rho*_{H^S}(chi_sigma^S(sigma_i)),
    where H^S = (V, E+ union S) and S ranges over subsets of E-.
    """
    variables = list(variables)
    if len(variables) > 5:
        raise WidthLimitError("sfhw order search is implemented for |V| <= 5")
    pos = [_as_edge(edge) for edge in pos_edges]
    neg = [_as_edge(edge) for edge in neg_edges]
    # Duplicate negative variable-sets do not change H^S.
    neg = list(dict.fromkeys(neg))
    if not variables:
        return 0.0
    best = math.inf
    for order in itertools.permutations(variables):
        width = _width_for_order(variables, pos, neg, order)
        if width < best:
            best = width
    return float(best)


def fhw_edges(variables, pos_edges):
    """Fractional hypertree width via the canonical elimination orders.

    On a positive query this equals sfhw (Lemma 3.9). The minimum is over
    variable orders, and each order is scored by the worst canonical bag.
    """
    return sfhw_edges(variables, pos_edges, [])


def witnessing_order(variables, pos_edges, neg_edges):
    """A variable order achieving sfhw, or any order if several tie."""
    variables = list(variables)
    if len(variables) > 5:
        raise WidthLimitError("sfhw order search is implemented for |V| <= 5")
    pos = [_as_edge(edge) for edge in pos_edges]
    neg = list(dict.fromkeys(_as_edge(edge) for edge in neg_edges))
    best_order = tuple(variables)
    best = math.inf
    for order in itertools.permutations(variables):
        width = _width_for_order(variables, pos, neg, order)
        if width < best:
            best = width
            best_order = order
    return list(best_order)
