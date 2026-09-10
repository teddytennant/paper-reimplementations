"""CPU identity checks for WRP (arXiv:2609.09883)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wrp.allocate import allocate_removals
from wrp.cluster import select_num_clusters, spectral_cluster
from wrp.constants import EPS, SCALE_KEYS
from wrp.prune import prune_blocks
from wrp.similarity import (
    center_output_channels,
    depth_binned_similarity,
    gram,
    linear_cka,
    pairwise_similarity,
    projection_scale_vectors,
)


def _random_block(key, d=8, m=12):
    ks = jax.random.split(key, 7)
    return {
        "q": jax.random.normal(ks[0], (d, d)),
        "k": jax.random.normal(ks[1], (d, d)),
        "v": jax.random.normal(ks[2], (d, d)),
        "o": jax.random.normal(ks[3], (d, d)),
        "gate": jax.random.normal(ks[4], (m, d)),
        "up": jax.random.normal(ks[5], (m, d)),
        "down": jax.random.normal(ks[6], (d, m)),
    }


def _random_weights(key, n_layers, d=8, m=12):
    keys = jax.random.split(key, n_layers)
    return [_random_block(k, d=d, m=m) for k in keys]


def _symmetrize_unit_diag(s):
    s = 0.5 * (s + s.T)
    n = s.shape[0]
    return s.at[jnp.diag_indices(n)].set(1.0)


def _block_similarity(membership, intra=0.95, inter=0.05):
    n = len(membership)
    labels = jnp.array(membership)
    same = labels[:, None] == labels[None, :]
    s = jnp.where(same, intra, inter).astype(jnp.float32)
    return _symmetrize_unit_diag(s)


def test_centering_zero_mean_across_output_dim():
    key = jax.random.PRNGKey(0)
    weight = jax.random.normal(key, (16, 11))
    centered = center_output_channels(weight)
    np.testing.assert_allclose(np.array(centered.mean(axis=0)), 0.0, atol=1e-6)


def test_gram_permutation_invariance():
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    weight = jax.random.normal(k1, (10, 7))
    perm = jax.random.permutation(k2, weight.shape[1])
    g = gram(weight)
    g_perm = gram(weight[:, perm])
    np.testing.assert_allclose(np.array(g), np.array(g_perm), atol=1e-5, rtol=1e-5)


def test_linear_cka_identical_and_numpy_reference():
    key = jax.random.PRNGKey(2)
    features = jax.random.normal(key, (9, 5))
    g = features @ features.T
    s_same = float(linear_cka(g, g))
    assert s_same == pytest.approx(1.0, abs=1e-6)

    k1, k2 = jax.random.split(key)
    g_a = gram(jax.random.normal(k1, (8, 6)))
    g_b = gram(jax.random.normal(k2, (8, 6)))
    s = float(linear_cka(g_a, g_b))
    g_a_np = np.array(g_a)
    g_b_np = np.array(g_b)
    ref = np.sum(g_a_np * g_b_np) / (
        np.linalg.norm(g_a_np) * np.linalg.norm(g_b_np) + EPS
    )
    assert s == pytest.approx(ref, abs=1e-6)
    evals_a = np.linalg.eigvalsh(g_a_np)
    evals_b = np.linalg.eigvalsh(g_b_np)
    assert evals_a.min() >= -1e-5
    assert evals_b.min() >= -1e-5
    assert 0.0 - 1e-6 <= s <= 1.0 + 1e-6


def test_similarity_unit_diagonal_and_symmetric():
    weights = _random_weights(jax.random.PRNGKey(3), n_layers=5, d=6, m=8)
    s = pairwise_similarity(weights)
    np.testing.assert_allclose(np.array(jnp.diag(s)), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.array(s), np.array(s.T), atol=1e-6)


def test_scale_vector_length_and_layer_centering():
    weights = _random_weights(jax.random.PRNGKey(4), n_layers=7, d=5, m=9)
    scale = projection_scale_vectors(weights)
    assert scale.shape == (7, len(SCALE_KEYS))
    centered = scale - jnp.mean(scale, axis=0, keepdims=True)
    np.testing.assert_allclose(np.array(centered.mean(axis=0)), 0.0, atol=1e-6)


def test_depth_binned_s0_constant_on_lags():
    weights = _random_weights(jax.random.PRNGKey(5), n_layers=6, d=6, m=7)
    s = pairwise_similarity(weights)
    s0 = depth_binned_similarity(s)
    n = s0.shape[0]
    np.testing.assert_allclose(np.array(jnp.diag(s0)), 1.0, atol=1e-6)
    s0_np = np.array(s0)
    for lag in range(1, n):
        vals = [s0_np[i, j] for i in range(n) for j in range(n) if abs(i - j) == lag]
        np.testing.assert_allclose(vals, vals[0], atol=1e-6)


def test_k_selection_two_four_and_degenerate():
    # Coarse two-block structure: d2 dominates the depth-binned reference.
    two_block = _block_similarity([0, 0, 0, 0, 1, 1, 1, 1], intra=0.98, inter=0.02)
    k2, rho2 = select_num_clusters(two_block)
    assert k2 == 2
    assert rho2 >= 0.5

    # Three-cluster structure: d3 = λ4-λ3 dominates, so the 2-gap does not
    # and Eq. 5 selects the finer K=4 candidate.
    three_block = _block_similarity(
        [0, 0, 0, 1, 1, 1, 2, 2, 2], intra=0.98, inter=0.02
    )
    k4, rho4 = select_num_clusters(three_block)
    assert k4 == 4
    assert rho4 < 0.5

    # S already depth-binned => S0 = S => e2 = e3 = 0 => K = 2, rho = 1.
    n = 6
    lags = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
    toeplitz = jnp.exp(-lags.astype(jnp.float32) / 2.0)
    toeplitz = _symmetrize_unit_diag(toeplitz)
    s0 = depth_binned_similarity(toeplitz)
    np.testing.assert_allclose(np.array(toeplitz), np.array(s0), atol=1e-6)
    k_deg, rho_deg = select_num_clusters(toeplitz)
    assert k_deg == 2
    assert rho_deg == pytest.approx(1.0, abs=1e-6)


def test_first_and_last_never_removed():
    for n_layers, p in ((6, 2), (8, 3), (10, 4)):
        weights = _random_weights(jax.random.PRNGKey(10 + n_layers), n_layers)
        removed = np.array(prune_blocks(weights, p))
        assert 0 not in removed
        assert (n_layers - 1) not in removed


def test_removal_cardinality_matches_p():
    for n_layers, p in ((5, 1), (6, 2), (8, 3), (7, 5), (6, 10)):
        weights = _random_weights(jax.random.PRNGKey(20 + n_layers + p), n_layers)
        removed = np.array(prune_blocks(weights, p))
        expected = min(p, n_layers - 2)
        assert len(removed) == expected
        assert len(set(removed.tolist())) == expected


def test_stage1_then_stage2_hand_built_allocation():
    """L=6, P=2, two clusters. Expected D is derived from the allocation rule.

    Clusters C0={0,1,2}, C1={3,4,5}. Protect 0 and 5.
    r(ell; C) = mean similarity to other members of C.

    r(0;C0)=(0.90+0.80)/2=0.85
    r(1;C0)=(0.90+0.95)/2=0.925
    r(2;C0)=(0.80+0.95)/2=0.875
    Eligible in C0: {1,2} -> remove 1.

    r(3;C1)=(0.70+0.75)/2=0.725
    r(4;C1)=(0.70+0.95)/2=0.825
    r(5;C1)=(0.75+0.95)/2=0.85
    Eligible in C1: {3,4} -> remove 4.

    Stage 1 already yields |D|=P=2, so D={1,4}.
    """
    s = jnp.array(
        [
            [1.00, 0.90, 0.80, 0.10, 0.10, 0.10],
            [0.90, 1.00, 0.95, 0.10, 0.10, 0.10],
            [0.80, 0.95, 1.00, 0.10, 0.10, 0.10],
            [0.10, 0.10, 0.10, 1.00, 0.70, 0.75],
            [0.10, 0.10, 0.10, 0.70, 1.00, 0.95],
            [0.10, 0.10, 0.10, 0.75, 0.95, 1.00],
        ],
        dtype=jnp.float32,
    )
    labels = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    removed = np.array(allocate_removals(s, labels, p=2))
    expected = np.array([1, 4], dtype=np.int32)
    np.testing.assert_array_equal(removed, expected)

    clustered = spectral_cluster(s, k=2, seed=0)
    # Clusters ordered by shallowest block: {0,1,2} then {3,4,5}.
    np.testing.assert_array_equal(np.array(clustered), np.array(labels))
    removed_from_clusters = np.array(allocate_removals(s, clustered, p=2))
    np.testing.assert_array_equal(removed_from_clusters, expected)
