"""CAST-SCHURTREE local tests: trees, weights, unbiasedness, order independence."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_cast.schur import (
    cast_schur_tree,
    exact_schur_clique,
    inverse_marginal_weight,
    is_connected,
    laplacian_from_edges,
    leverage_scores,
)


def _sample_many(a, rho, n_samples, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), n_samples)
    return [cast_schur_tree(a, rho, k) for k in keys]


def test_d_le_1_empty():
    key = jax.random.PRNGKey(0)
    assert cast_schur_tree(jnp.array([]), 1, key) == []
    assert cast_schur_tree(jnp.array([1.5]), 1, key) == []
    assert cast_schur_tree(jnp.array([1.5]), 4, key) == []


def test_cast1_tree_size_and_connected():
    a = jnp.array([0.5, 1.0, 1.5, 2.5])
    d = int(a.size)
    for i, edges in enumerate(_sample_many(a, 1, 40, seed=1)):
        assert len(edges) == d - 1, i
        assert is_connected(d, edges)


def test_cast1_selected_edge_conductance():
    a = np.array([0.5, 1.0, 1.5, 2.5], dtype=np.float64)
    for edges in _sample_many(jnp.asarray(a), 1, 30, seed=2):
        assert len(edges) == 3
        for i, j, h in edges:
            want = inverse_marginal_weight(a[i], a[j], rho=1)
            assert abs(h - want) < 1e-6
            assert abs(h - a[i] * a[j] / (a[i] + a[j])) < 1e-6


def test_cast1_unbiased_schur():
    a = jnp.array([0.5, 1.0, 1.5, 2.5])
    d = int(a.size)
    exact = exact_schur_clique(a)
    acc = np.zeros((d, d), dtype=np.float64)
    samples = _sample_many(a, 1, 1500, seed=3)
    for edges in samples:
        acc += laplacian_from_edges(d, edges)
    mean = acc / len(samples)
    assert np.allclose(mean, exact, atol=0.05, rtol=0.0)


def test_cast2_unbiased_schur():
    a = jnp.array([0.5, 1.0, 1.5, 2.5])
    d = int(a.size)
    exact = exact_schur_clique(a)
    acc = np.zeros((d, d), dtype=np.float64)
    samples = _sample_many(a, 2, 1500, seed=4)
    for edges in samples:
        acc += laplacian_from_edges(d, edges)
        assert is_connected(d, edges)
        assert len(edges) <= 2 * d - 1
    mean = acc / len(samples)
    assert np.allclose(mean, exact, atol=0.08, rtol=0.0)


def test_cast1_inclusion_matches_leverage():
    a = jnp.array([0.5, 1.0, 1.5, 2.5])
    d = int(a.size)
    tau = leverage_scores(a)
    n_samples = 2000
    freq = np.zeros((d, d), dtype=np.float64)
    for edges in _sample_many(a, 1, n_samples, seed=5):
        seen = set()
        for i, j, _h in edges:
            seen.add((min(i, j), max(i, j)))
        for i, j in seen:
            freq[i, j] += 1.0
            freq[j, i] += 1.0
    freq /= n_samples
    iu = np.triu_indices(d, k=1)
    assert np.allclose(freq[iu], tau[iu], atol=0.05, rtol=0.0)


def test_local_order_independence():
    a = np.array([0.5, 1.0, 1.5, 2.5], dtype=np.float64)
    perm = np.array([2, 0, 3, 1])
    a_perm = a[perm]

    n_samples = 2000
    d = a.size

    def pair_freq(vec, seed):
        freq = np.zeros((d, d), dtype=np.float64)
        for edges in _sample_many(jnp.asarray(vec), 1, n_samples, seed=seed):
            for i, j, _h in edges:
                freq[i, j] += 1.0
                freq[j, i] += 1.0
        return freq / n_samples

    f_id = pair_freq(a, seed=6)
    f_p = pair_freq(a_perm, seed=7)
    remapped = np.zeros_like(f_p)
    for i in range(d):
        for j in range(d):
            remapped[perm[i], perm[j]] = f_p[i, j]
    iu = np.triu_indices(d, k=1)
    assert np.allclose(f_id[iu], remapped[iu], atol=0.06, rtol=0.0)
