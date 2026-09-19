"""Algorithm 1: one-sided coordinate-wise monotonicity tester."""

from __future__ import annotations

import math

import numpy as np

from ic_tester import (
    ALG1_C,
    default_alg1_repeats,
    coordinatewise_monotonicity,
)


def _min_clip(d: int):
    def x(bids):
        bids = np.asarray(bids)
        return np.minimum(d, bids)

    return x


def _threshold(thresh: int, d: int = 1):
    def x(bids):
        bids = np.asarray(bids)
        return np.where(bids >= thresh, d, 0)

    return x


def _decreasing_coord0(m: int, n: int):
    def x(bids):
        bids = np.asarray(bids, dtype=int)
        out = np.minimum(bids, m).copy()
        out[0] = m - int(bids[0])
        return out

    return x


def test_alg1_never_rejects_monotone():
    n, m, d, eps = 3, 8, 4, 0.2
    rng = np.random.default_rng(0)
    for kind in ("clip", "threshold"):
        x = _min_clip(d) if kind == "clip" else _threshold(3, d=1)
        for run in range(30):
            result = coordinatewise_monotonicity(
                x, n=n, m=m, eps=eps, rng=rng
            )
            assert result.accepted, f"{kind} run {run} rejected"


def test_alg1_rejects_far_from_monotone():
    n, m, eps = 2, 8, 0.25
    x = _decreasing_coord0(m, n)
    rng = np.random.default_rng(1)
    n_runs = 10
    n_reject = 0
    for _ in range(n_runs):
        result = coordinatewise_monotonicity(x, n=n, m=m, eps=eps, rng=rng)
        if not result.accepted:
            n_reject += 1
    assert n_reject >= 1
    assert n_reject >= n_runs / 2


def test_alg1_query_count_and_repeats():
    n, m, eps = 3, 8, 0.2
    x = _min_clip(4)
    rng = np.random.default_rng(2)
    result = coordinatewise_monotonicity(x, n=n, m=m, eps=eps, rng=rng)
    expected = default_alg1_repeats(n, m, eps)
    assert expected == math.ceil(ALG1_C * n * math.log(m) / eps)
    assert result.n_repeats == expected
    assert result.n_queries <= 2 * result.n_repeats
    # Accepted monotone run uses the full budget, two queries per trial.
    assert result.accepted
    assert result.n_queries == 2 * result.n_repeats
