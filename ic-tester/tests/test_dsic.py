"""Algorithms 2–3 and 1D binary search."""

from __future__ import annotations

import numpy as np

from ic_tester import (
    binary_search_first,
    myerson_price_1d,
    dsic,
    dsic_1d,
)


def _table_mechanism(x: np.ndarray, p: np.ndarray):
    def a(b: int):
        b = int(b)
        return int(x[b]), float(p[b])

    return a


def test_alg2_accepts_dsic_pair():
    x = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    p = myerson_price_1d(x)
    a = _table_mechanism(x, p)
    rng = np.random.default_rng(0)
    m = int(x.shape[0]) - 1
    for _ in range(20):
        result = dsic_1d(a, m=m, d=2, eps=0.25, rng=rng)
        assert result.accepted


def test_alg2_rejects_t0_and_bad_h():
    rng = np.random.default_rng(1)
    # x(0)=0, x(b)=1 for b>=1, but p too large for any h in [0, 1].
    x = np.array([0, 1, 1, 1, 1])
    p = np.array([0.0, 5.0, 5.0, 5.0, 5.0])
    result = dsic_1d(_table_mechanism(x, p), m=4, d=1, eps=0.5, rng=rng)
    assert not result.accepted

    # x(0)=1 and a later 0: binary search reports t0 != 0.
    x2 = np.array([1, 0, 0, 0])
    p2 = np.zeros(4)
    result2 = dsic_1d(_table_mechanism(x2, p2), m=3, d=1, eps=0.5, rng=rng)
    assert not result2.accepted


def test_alg2_rejects_price_change_on_plateau():
    x = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    p = myerson_price_1d(x).copy()
    # Valid left-endpoint price is constant on each plateau; break the
    # second plateau.
    p[7] = p[4] + 1.0
    p[8] = p[4] + 2.0
    a = _table_mechanism(x, p)
    rng = np.random.default_rng(2)
    n_runs = 10
    n_reject = 0
    for _ in range(n_runs):
        result = dsic_1d(a, m=8, d=1, eps=0.5, rng=rng)
        if not result.accepted:
            n_reject += 1
    assert n_reject >= 1
    assert n_reject >= n_runs // 2


def test_alg3_accepts_dsic_posted_price():
    n, m, d, eps = 2, 4, 1, 0.5
    thresh = 2

    def a(bids):
        bids = np.asarray(bids, dtype=int)
        x = (bids >= thresh).astype(int)
        p = np.where(x, float(thresh - 1), 0.0)
        return x, p

    rng = np.random.default_rng(3)
    for _ in range(3):
        result = dsic(a, n=n, m=m, d=d, eps=eps, rng=rng)
        assert result.accepted


def test_binary_search_first_on_monotone():
    x = [0, 0, 0, 1, 1, 2, 2, 2, 4]
    m = len(x) - 1

    def x_fn(b: int) -> int:
        return x[b]

    assert binary_search_first(x_fn, m, 0) == 0
    assert binary_search_first(x_fn, m, 1) == 3
    assert binary_search_first(x_fn, m, 2) == 5
    assert binary_search_first(x_fn, m, 3) is None
    assert binary_search_first(x_fn, m, 4) == 8
    # Constant allocation.
    const = [2, 2, 2, 2]

    def c_fn(b: int) -> int:
        return const[b]

    assert binary_search_first(c_fn, 3, 2) == 0
    assert binary_search_first(c_fn, 3, 0) is None
    assert binary_search_first(c_fn, 3, 1) is None
