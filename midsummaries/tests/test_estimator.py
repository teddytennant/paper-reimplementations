import math

import numpy as np
import pytest

from midsummaries import (
    estimate,
    estimate_crossfit,
    jp,
    mid_summary,
    weights_and_mids,
)


def test_mids_and_weights_on_handcrafted_unique_values():
    x = np.arange(64, dtype=np.float64)
    k = 2
    lam, mids, ell = weights_and_mids(x, k)
    m = 32
    train = np.sort(x[:m])
    J = jp(64, k)
    assert len(mids) == J + 1
    assert len(lam) == J + 1
    assert len(ell) == J + 1
    for j in range(J + 1):
        r = (2**j) * k
        expected = (train[r - 1] + train[m - r]) / 2.0
        assert abs(float(mids[j]) - expected) < 1e-12
        assert abs(float(mids[j]) - mid_summary(train, r)) < 1e-12
    lam_np = np.asarray(lam, dtype=np.float64)
    assert np.all(lam_np >= 0.0)
    assert abs(float(lam_np.sum()) - 1.0) < 1e-12


def test_translation_equivariance():
    x = np.array(
        [0.3, -1.2, 4.0, 2.2, 5.5, -0.7, 1.1, 3.3, 8.0, 0.0, -2.5, 6.1],
        dtype=np.float64,
    )
    c = 12.5
    assert abs(estimate(x + c) - estimate(x) - c) < 1e-10
    assert abs(estimate_crossfit(x + c) - estimate_crossfit(x) - c) < 1e-10


def test_odd_symmetry():
    x = np.array(
        [0.4, -1.7, 3.2, 2.8, 5.1, -0.9, 1.3, 4.6, 8.2, 0.2, -2.4, 6.5],
        dtype=np.float64,
    )
    assert abs(estimate(-x) + estimate(x)) < 1e-10
    assert abs(estimate_crossfit(-x) + estimate_crossfit(x)) < 1e-10


def test_jp_formula_n64_k2():
    n, k = 64, 2
    m = n // 2
    p = k / (m + 1)
    assert m == 32
    assert abs(p - 2 / 33) < 1e-15
    assert jp(n, k) == int(math.floor(math.log2(1.0 / (4.0 * p))))


def test_number_of_mids_is_log_n():
    n = 2**12
    x = np.arange(n, dtype=np.float64)
    lam, mids, ell = weights_and_mids(x, k=2)
    assert len(mids) == jp(n, 2) + 1
    assert len(mids) <= 20
    assert len(lam) == len(mids) == len(ell)


def test_gaussian_crossfit_near_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 1.0, size=400)
    mu = estimate_crossfit(x, k=2)
    assert abs(mu - 3.0) < 0.25


def test_uniform_crossfit_near_zero():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, size=800)
    mu = estimate_crossfit(x, k=2)
    assert abs(mu) < 0.08


def test_k_too_large_raises():
    x = np.arange(20, dtype=np.float64)
    with pytest.raises(ValueError):
        estimate(x, k=6)


def test_n_too_small_raises():
    x = np.arange(7, dtype=np.float64)
    with pytest.raises(ValueError):
        estimate(x, k=1)


def test_all_ties_return_the_common_value():
    assert estimate(np.ones(20)) == 1.0
    assert estimate_crossfit(np.full(16, 3.14)) == 3.14


def test_crossfit_averages_both_split_directions():
    x = np.array(
        [1.0, 4.0, 2.0, 9.0, 3.0, 8.0, 0.0, 5.0, 7.0, 6.0, -1.0, 11.0],
        dtype=np.float64,
    )
    m = len(x) // 2
    est_ab = estimate(x, k=2)
    est_ba = estimate(np.concatenate([x[m:], x[:m]]), k=2)
    got = estimate_crossfit(x, k=2)
    assert abs(got - 0.5 * (est_ab + est_ba)) < 1e-12
    assert abs(est_ab - est_ba) > 1e-12
