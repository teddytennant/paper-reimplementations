"""The decoder has to be exactly optimal, not approximately."""

import itertools

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from hllm.hungarian import (
    assign_positions,
    assignment_score,
    decode_ranking,
    solve_assignment,
)


def brute_force(cost, maximize=False):
    """Optimal value by enumerating every injective row-to-column map."""
    n, m = cost.shape
    if n > m:
        return brute_force(cost.T, maximize=maximize)
    values = [sum(cost[i, c[i]] for i in range(n)) for c in itertools.permutations(range(m), n)]
    return max(values) if maximize else min(values)


def check_valid(cost, rows, cols):
    n, m = cost.shape
    assert len(rows) == len(cols) == min(n, m)
    assert len(set(rows.tolist())) == len(rows), "rows repeat, not a matching"
    assert len(set(cols.tolist())) == len(cols), "columns repeat, not a matching"
    assert set(rows.tolist()) <= set(range(n))
    assert set(cols.tolist()) <= set(range(m))
    assert (np.diff(rows) > 0).all(), "rows should come out sorted"


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
def test_matches_brute_force_square(n):
    rng = np.random.default_rng(n)
    trials = 200 if n <= 5 else 25
    for _ in range(trials):
        cost = rng.normal(size=(n, n))
        rows, cols = solve_assignment(cost)
        check_valid(cost, rows, cols)
        assert cost[rows, cols].sum() == pytest.approx(brute_force(cost), abs=1e-9)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_matches_brute_force_with_ties_and_negatives(n):
    """Integer matrices from a tiny range so ties and duplicates are everywhere."""
    rng = np.random.default_rng(1000 + n)
    for _ in range(150):
        cost = rng.integers(-3, 4, size=(n, n)).astype(float)
        rows, cols = solve_assignment(cost)
        check_valid(cost, rows, cols)
        assert cost[rows, cols].sum() == pytest.approx(brute_force(cost), abs=1e-9)


@pytest.mark.parametrize("shape", [(2, 5), (3, 7), (4, 4), (5, 2), (7, 3), (1, 6), (6, 1)])
def test_matches_brute_force_rectangular(shape):
    rng = np.random.default_rng(sum(shape) * 31)
    for _ in range(60):
        cost = rng.normal(size=shape)
        rows, cols = solve_assignment(cost)
        check_valid(cost, rows, cols)
        assert cost[rows, cols].sum() == pytest.approx(brute_force(cost), abs=1e-9)


@pytest.mark.parametrize("n", [3, 5, 6])
def test_maximize_matches_brute_force(n):
    """Equation (1) is a maximum, so this is the branch the decoder uses."""
    rng = np.random.default_rng(50 + n)
    for _ in range(100):
        scores = rng.normal(size=(n, n))
        rows, cols = solve_assignment(scores, maximize=True)
        assert scores[rows, cols].sum() == pytest.approx(
            brute_force(scores, maximize=True), abs=1e-9
        )


@pytest.mark.parametrize("shape", [(20, 20), (50, 50), (97, 97), (17, 40), (40, 17), (64, 65)])
def test_matches_scipy(shape):
    rng = np.random.default_rng(sum(shape))
    for _ in range(10):
        cost = rng.normal(size=shape) * rng.integers(1, 100)
        rows, cols = solve_assignment(cost)
        check_valid(cost, rows, cols)
        ref_r, ref_c = linear_sum_assignment(cost)
        assert cost[rows, cols].sum() == pytest.approx(cost[ref_r, ref_c].sum(), rel=1e-12)


def test_matches_scipy_on_degenerate_matrices():
    rng = np.random.default_rng(7)
    cases = [
        np.zeros((12, 12)),
        np.ones((12, 12)),
        np.full((9, 14), -4.0),
        rng.integers(0, 2, size=(30, 30)).astype(float),
        np.tile(rng.normal(size=(1, 25)), (25, 1)),  # every row identical
        np.tile(rng.normal(size=(25, 1)), (1, 25)),  # every column identical
    ]
    for cost in cases:
        rows, cols = solve_assignment(cost)
        check_valid(cost, rows, cols)
        ref_r, ref_c = linear_sum_assignment(cost)
        assert cost[rows, cols].sum() == pytest.approx(cost[ref_r, ref_c].sum(), abs=1e-9)


def test_diagonal_matrix_decodes_to_identity():
    """A strongly diagonal M means item i belongs at position i."""
    for n in (1, 4, 12, 40):
        scores = np.eye(n) * 5.0 + np.random.default_rng(n).normal(size=(n, n)) * 0.1
        assert np.array_equal(decode_ranking(scores), np.arange(n))
        assert np.array_equal(assign_positions(scores), np.arange(n))


def test_decoded_ranking_is_always_a_permutation():
    """Section 6.2: valid by construction, over garbage input too."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 25))
        scores = rng.normal(size=(n, n)) * rng.choice([1e-6, 1.0, 1e6])
        order = decode_ranking(scores)
        assert np.array_equal(np.sort(order), np.arange(n))


def test_assign_positions_agrees_with_decode_ranking():
    rng = np.random.default_rng(3)
    for shape in [(6, 6), (4, 9), (9, 4)]:
        scores = rng.normal(size=shape)
        pos = assign_positions(scores)
        order = decode_ranking(scores)
        assert len(order) == min(shape)
        used = pos[order]
        assert (np.diff(used) > 0).all(), "decode_ranking must sort by position"
        assert (pos >= 0).sum() == min(shape)
        assert sorted(used.tolist()) == sorted(pos[pos >= 0].tolist())
        assert assignment_score(scores, pos) == pytest.approx(scores[order, used].sum())


def test_row_and_column_shifts_leave_the_assignment_alone():
    """Every row and every column is used exactly once in the square case, so
    adding a constant to one of them shifts every feasible objective by the
    same amount and cannot move the argmax."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        scores = rng.normal(size=(9, 9))
        base = decode_ranking(scores)
        assert np.array_equal(decode_ranking(scores + rng.normal(size=(9, 1))), base)
        assert np.array_equal(decode_ranking(scores + rng.normal(size=(1, 9))), base)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        solve_assignment(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        solve_assignment(np.array([[1.0, np.inf], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        solve_assignment(np.array([[1.0, np.nan], [0.0, 1.0]]))
    rows, cols = solve_assignment(np.zeros((0, 4)))
    assert len(rows) == len(cols) == 0
