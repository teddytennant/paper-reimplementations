"""The Hellinger distance itself, against closed forms and its metric axioms.

Section 2.3 defines the contextual dissimilarity as Gamma_h = H^2, so every
later bound inherits whatever this function does.
"""

import numpy as np
import pytest

from tokengeom.hellinger import (
    bhattacharyya_coefficient,
    hellinger,
    pairwise_hellinger,
    pairwise_squared_hellinger,
    squared_hellinger,
)


def random_simplex(rng, n, k):
    x = rng.random((n, k)) + 1e-3
    return x / x.sum(axis=1, keepdims=True)


def test_bernoulli_closed_form():
    """H^2(Bern(p), Bern(q)) = 1 - sqrt(pq) - sqrt((1-p)(1-q))."""
    rng = np.random.default_rng(0)
    for p, q in rng.random((50, 2)):
        got = float(squared_hellinger([p, 1 - p], [q, 1 - q]))
        want = 1.0 - np.sqrt(p * q) - np.sqrt((1 - p) * (1 - q))
        assert got == pytest.approx(want, abs=1e-14)


def test_binomial_closed_form():
    """H^2(Bin(n,p), Bin(n,q)) = 1 - (sqrt(pq) + sqrt((1-p)(1-q)))^n.

    A product measure, so the Bhattacharyya coefficient is the n-th power of the
    single-trial one. Nontrivial because the two pmfs share full support.
    """
    from math import comb

    rng = np.random.default_rng(1)
    for n in (1, 3, 8, 20):
        for p, q in rng.random((10, 2)):
            support = np.arange(n + 1)
            coef = np.array([comb(n, int(k)) for k in support], dtype=np.float64)
            pk = coef * p**support * (1 - p) ** (n - support)
            qk = coef * q**support * (1 - q) ** (n - support)
            want = 1.0 - (np.sqrt(p * q) + np.sqrt((1 - p) * (1 - q))) ** n
            assert float(squared_hellinger(pk, qk)) == pytest.approx(want, abs=1e-12)


def test_geometric_closed_form():
    """Truncated geometrics against sum_k sqrt(p q) (1-p)^{k/2} (1-q)^{k/2}."""
    p, q, n = 0.3, 0.55, 400
    k = np.arange(n)
    pk = p * (1 - p) ** k
    qk = q * (1 - q) ** k
    ratio = np.sqrt((1 - p) * (1 - q))
    want = np.sqrt(p * q) * (1 - ratio**n) / (1 - ratio)
    assert float(bhattacharyya_coefficient(pk, qk)) == pytest.approx(want, abs=1e-12)


def test_disjoint_support_and_identity():
    a = np.array([0.5, 0.5, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.25, 0.75])
    assert float(squared_hellinger(a, b)) == pytest.approx(1.0, abs=1e-15)
    assert float(squared_hellinger(a, a)) == pytest.approx(0.0, abs=1e-15)


def test_range_symmetry_and_identity_of_indiscernibles():
    rng = np.random.default_rng(2)
    p = random_simplex(rng, 200, 7)
    q = random_simplex(rng, 200, 7)
    h2 = np.asarray(squared_hellinger(p, q))
    assert h2.min() >= 0.0
    assert h2.max() <= 1.0
    assert np.allclose(h2, np.asarray(squared_hellinger(q, p)), atol=1e-15)
    assert np.all(h2 > 0)
    assert np.allclose(np.asarray(squared_hellinger(p, p)), 0.0, atol=1e-15)


def test_triangle_inequality():
    """H is a metric, which is what lets Section 2.3 transfer the model-implied
    geometry to the population one."""
    rng = np.random.default_rng(3)
    p, q, r = (random_simplex(rng, 500, 6) for _ in range(3))
    hpq = np.asarray(hellinger(p, q))
    hqr = np.asarray(hellinger(q, r))
    hpr = np.asarray(hellinger(p, r))
    assert np.all(hpr <= hpq + hqr + 1e-12)


def test_pairwise_matches_elementwise():
    rng = np.random.default_rng(4)
    p = random_simplex(rng, 12, 9)
    m = np.asarray(pairwise_squared_hellinger(p))
    for i in range(12):
        for j in range(12):
            assert m[i, j] == pytest.approx(float(squared_hellinger(p[i], p[j])), abs=1e-14)
    assert np.allclose(np.asarray(pairwise_hellinger(p)) ** 2, m, atol=1e-14)
