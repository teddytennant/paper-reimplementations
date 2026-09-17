"""Numerical checks of ME-Decoding (arXiv:2609.18723) on synthetic tensors."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_me_decoding import (
    adaptive_bandwidth,
    greedy_select,
    mahalanobis_energy,
    me_sample,
    me_select,
    me_truncate_logits,
    rbf_kernel,
)


def _softmax(z, t=1.0):
    return jax.nn.softmax(jnp.asarray(z) / t)


# --- 1. Bandwidth -------------------------------------------------------------


def test_bandwidth_two_point_is_the_single_distance():
    embeddings = jnp.array([[0.0, 0.0], [3.0, 4.0]])
    eps = adaptive_bandwidth(embeddings)
    np.testing.assert_allclose(float(eps), 5.0, atol=1e-6)


def test_bandwidth_identical_is_zero():
    embeddings = jnp.ones((4, 3))
    eps = adaptive_bandwidth(embeddings)
    np.testing.assert_allclose(float(eps), 0.0, atol=1e-6)


def test_bandwidth_single_token_is_zero():
    embeddings = jnp.array([[1.0, 2.0, 3.0]])
    eps = adaptive_bandwidth(embeddings)
    np.testing.assert_allclose(float(eps), 0.0, atol=1e-6)


# --- 2. Kernel ----------------------------------------------------------------


def test_kernel_identical_is_ones():
    embeddings = jnp.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    eps = adaptive_bandwidth(embeddings)
    k = rbf_kernel(embeddings, eps)
    np.testing.assert_allclose(k, jnp.ones((3, 3)), atol=1e-6)


def test_kernel_orthogonal_offdiag_lt_one_and_psd():
    embeddings = jnp.eye(4)
    eps = adaptive_bandwidth(embeddings)
    k = rbf_kernel(embeddings, eps)
    off = k - jnp.diag(jnp.diag(k))
    assert float(jnp.max(off)) < 1.0 - 1e-6
    np.testing.assert_allclose(k, k.T, atol=1e-6)
    eig = jnp.linalg.eigvalsh(k)
    assert float(jnp.min(eig)) >= -1e-5


def test_kernel_epsilon_zero_is_ones_even_if_points_differ():
    embeddings = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    k = rbf_kernel(embeddings, 0.0)
    np.testing.assert_allclose(k, jnp.ones((3, 3)), atol=1e-6)


# --- 3. First token -----------------------------------------------------------


def test_first_token_when_k_is_identity_is_argmax_p_sq():
    p = jnp.array([0.1, 0.55, 0.2, 0.15])
    k = jnp.eye(4)
    selected, _, info = greedy_select(p, k, lambda_=0.9, method="cholesky")
    assert int(selected[0]) == 1
    assert info["first_token"] == 1
    scores = (p * p) / jnp.diag(k)
    assert int(selected[0]) == int(jnp.argmax(scores))


def test_first_token_matches_reference():
    p = jnp.array([0.05, 0.4, 0.35, 0.2])
    k = jnp.eye(4)
    sel_c, _, _ = greedy_select(p, k, lambda_=1.0, method="cholesky")
    sel_r, _, _ = greedy_select(p, k, lambda_=1.0, method="reference")
    assert int(sel_c[0]) == int(sel_r[0]) == int(jnp.argmax(p * p))


# --- 4. Duplicate embeddings (redundancy discount) ----------------------------


def test_duplicate_embeddings_not_both_selected():
    embeddings = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    logits = jnp.array([3.0, 2.8, 0.5])
    selected, f_path, info = me_select(
        logits, embeddings, lambda_=0.01, temperature=1.0, top_n=3
    )
    sel = {int(i) for i in np.asarray(selected)}
    assert (0 in sel) ^ (1 in sel), sel
    assert 2 in sel, sel
    assert jnp.all(jnp.isfinite(f_path))
    assert not bool(jnp.any(jnp.isnan(info["epsilon"])))


# --- 5. Orthogonal high-p tokens ---------------------------------------------


def test_orthogonal_high_p_both_selectable_small_lambda():
    embeddings = jnp.eye(3)
    logits = jnp.array([2.0, 2.0, -8.0])
    selected, _, _ = me_select(
        logits, embeddings, lambda_=0.0, temperature=1.0, top_n=3
    )
    sel = {int(i) for i in np.asarray(selected)}
    assert 0 in sel and 1 in sel, sel


# --- 6. F monotone ------------------------------------------------------------


def test_f_monotone_nondecreasing_along_greedy_path():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    embeddings = jax.random.normal(k1, (8, 4))
    logits = jax.random.normal(k2, (8,))
    _, f_path, _ = me_select(logits, embeddings, lambda_=0.0, top_n=8)
    diffs = jnp.diff(f_path)
    assert jnp.all(diffs >= -1e-5), diffs


def test_f_monotone_with_early_stop():
    embeddings = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
    logits = jnp.arange(6.0)
    _, f_path, _ = me_select(logits, embeddings, lambda_=0.9, top_n=6)
    assert f_path.size >= 1
    if f_path.size > 1:
        assert jnp.all(jnp.diff(f_path) >= -1e-5)


# --- 7. Early stop ------------------------------------------------------------


def test_early_stop_large_lambda_selects_few_small_lambda_selects_more():
    # Peaked p on K = I: F(S) = ||p_S||^2, so the relative-gain rule is isolated.
    p = jnp.array([0.80, 0.10, 0.05, 0.03, 0.02])
    k = jnp.eye(5)
    sel_hi, _, _ = greedy_select(p, k, lambda_=1.0, method="cholesky")
    sel_mid, _, _ = greedy_select(p, k, lambda_=0.99, method="cholesky")
    sel_lo, _, _ = greedy_select(p, k, lambda_=0.0, method="cholesky")
    assert len(sel_hi) < len(sel_lo)
    assert len(sel_mid) <= len(sel_lo)
    assert len(sel_hi) <= 2
    assert len(sel_lo) >= 4

    # Same qualitative behaviour through the public API (peaked logits).
    embeddings = jnp.stack(
        [
            jnp.array([1.0, 0.0]),
            jnp.array([0.97, 0.05]),
            jnp.array([0.96, 0.04]),
            jnp.array([0.95, 0.06]),
            jnp.array([0.94, 0.03]),
        ]
    )
    logits = jnp.array([6.0, 1.0, 0.5, 0.2, 0.0])
    api_hi, _, _ = me_select(logits, embeddings, lambda_=1.0, top_n=5)
    api_lo, _, _ = me_select(logits, embeddings, lambda_=0.0, top_n=5)
    assert len(api_hi) <= len(api_lo)


# --- 8. Tiny-N brute force / rank-1 vs scratch --------------------------------


def test_tiny_n_first_step_matches_exhaustive_and_rank1_matches_scratch():
    key = jax.random.PRNGKey(2)
    n, d = 5, 3
    embeddings = jax.random.normal(key, (n, d))
    logits = jnp.array([0.4, 1.1, -0.3, 0.8, 0.2])
    p = _softmax(logits)
    eps = adaptive_bandwidth(embeddings)
    k = rbf_kernel(embeddings, eps)

    singleton_f = [(j, float(p[j] ** 2 / k[j, j])) for j in range(n)]
    best_j = max(singleton_f, key=lambda t: t[1])[0]

    sel_c, f_c, _ = greedy_select(p, k, lambda_=0.0, method="cholesky")
    sel_r, f_r, _ = greedy_select(p, k, lambda_=0.0, method="reference")
    assert int(sel_c[0]) == best_j
    assert int(sel_r[0]) == best_j
    np.testing.assert_allclose(np.asarray(f_c), np.asarray(f_r), atol=1e-5)

    sel_list = [int(i) for i in np.asarray(sel_c)]
    for t in range(len(sel_list)):
        prefix = sel_list[: t + 1]
        idx = jnp.asarray(prefix)
        f_scratch = mahalanobis_energy(p[idx], k[idx[:, None], idx])
        np.testing.assert_allclose(float(f_c[t]), float(f_scratch), atol=1e-5)


def test_tiny_n_greedy_vs_exhaustive_best_subset_same_size():
    embeddings = jax.random.normal(jax.random.PRNGKey(3), (4, 3))
    logits = jnp.array([1.5, 0.2, 0.9, -0.4])
    p = _softmax(logits)
    k = rbf_kernel(embeddings, adaptive_bandwidth(embeddings))
    sel, f_path, _ = greedy_select(p, k, lambda_=0.0, method="cholesky")
    sel_list = [int(i) for i in np.asarray(sel)]

    def f_of(subset):
        idx = jnp.asarray(list(subset))
        return float(mahalanobis_energy(p[idx], k[idx[:, None], idx]))

    # size 1: greedy must match exhaustive
    best1 = max(range(4), key=lambda j: f_of([j]))
    assert sel_list[0] == best1

    for m in range(1, len(sel_list) + 1):
        greedy_f = float(f_path[m - 1])
        best_f = max(f_of(s) for s in itertools.combinations(range(4), m))
        assert greedy_f <= best_f + 1e-5
        np.testing.assert_allclose(greedy_f, f_of(sel_list[:m]), atol=1e-5)


def test_rank1_matches_scratch_on_identity_kernel():
    p = jnp.array([0.5, 0.3, 0.15, 0.05])
    k = jnp.eye(4)
    sel, f_c, _ = greedy_select(p, k, lambda_=0.0, method="cholesky")
    _, f_r, _ = greedy_select(p, k, lambda_=0.0, method="reference")
    np.testing.assert_allclose(np.asarray(f_c), np.asarray(f_r), atol=1e-5)
    prefix = []
    for t, j in enumerate(np.asarray(sel)):
        prefix.append(int(j))
        idx = jnp.asarray(prefix)
        f_scratch = mahalanobis_energy(p[idx], k[idx[:, None], idx])
        np.testing.assert_allclose(float(f_c[t]), float(f_scratch), atol=1e-5)


# --- 9. Truncated logits ------------------------------------------------------


def test_truncated_logits_unselected_neg_inf_selected_equal_softmax_on_s():
    embeddings = jnp.eye(5)
    logits = jnp.array([2.0, 1.0, 0.5, -1.0, -2.0])
    selected, _, _ = me_select(logits, embeddings, lambda_=0.5, top_n=5)
    trunc = me_truncate_logits(logits, embeddings, lambda_=0.5, top_n=5)
    sel = {int(i) for i in np.asarray(selected)}
    for i in range(5):
        if i in sel:
            np.testing.assert_allclose(float(trunc[i]), float(logits[i]), atol=1e-6)
        else:
            assert bool(jnp.isneginf(trunc[i]))
    probs = jax.nn.softmax(trunc)
    np.testing.assert_allclose(float(probs.sum()), 1.0, atol=1e-5)
    mass_s = float(sum(probs[i] for i in sel))
    np.testing.assert_allclose(mass_s, 1.0, atol=1e-5)


# --- 10. Sampling -------------------------------------------------------------


def test_sampling_returns_index_in_s():
    embeddings = jnp.eye(5)
    logits = jnp.array([3.0, 2.0, 0.0, -1.0, -4.0])
    selected, _, _ = me_select(logits, embeddings, lambda_=0.5, top_n=5)
    sel = {int(i) for i in np.asarray(selected)}
    key = jax.random.PRNGKey(0)
    for i in range(8):
        key, sub = jax.random.split(key)
        tok = int(me_sample(sub, logits, embeddings, lambda_=0.5, top_n=5))
        assert tok in sel


# --- 11. Temperature ----------------------------------------------------------


def test_temperature_towards_zero_first_token_is_argmax_logit():
    embeddings = jnp.ones((4, 3))
    logits = jnp.array([1.0, 5.0, 2.0, 0.0])
    selected, _, info = me_select(
        logits, embeddings, lambda_=0.9, temperature=1e-6, top_n=4
    )
    assert int(selected[0]) == 1
    np.testing.assert_allclose(float(info["epsilon"]), 0.0, atol=1e-6)


# --- 12. Degenerate inputs, no NaNs ------------------------------------------


def test_no_nans_all_embeddings_equal():
    embeddings = jnp.ones((5, 2)) * 3.0
    logits = jnp.array([0.2, 1.5, -0.3, 0.0, 0.4])
    selected, f_path, info = me_select(logits, embeddings, lambda_=0.9, top_n=5)
    assert selected.size >= 1
    assert jnp.all(jnp.isfinite(f_path))
    assert not bool(jnp.any(jnp.isnan(rbf_kernel(embeddings, info["epsilon"]))))
    trunc = me_truncate_logits(logits, embeddings, lambda_=0.9, top_n=5)
    assert not bool(jnp.any(jnp.isnan(trunc)))


def test_no_nans_v1():
    embeddings = jnp.array([[0.3, -1.0, 2.0]])
    logits = jnp.array([4.2])
    selected, f_path, info = me_select(logits, embeddings, lambda_=0.9, top_n=512)
    assert int(selected[0]) == 0
    assert selected.size == 1
    assert jnp.all(jnp.isfinite(f_path))
    assert not bool(jnp.isnan(info["epsilon"]))
    trunc = me_truncate_logits(logits, embeddings, top_n=512)
    np.testing.assert_allclose(float(trunc[0]), 4.2, atol=1e-6)
    tok = int(me_sample(jax.random.PRNGKey(1), logits, embeddings, top_n=512))
    assert tok == 0


def test_no_nans_top_n_greater_than_v():
    embeddings = jax.random.normal(jax.random.PRNGKey(4), (3, 5))
    logits = jnp.array([0.0, 1.0, -1.0])
    selected, f_path, info = me_select(logits, embeddings, lambda_=0.5, top_n=10_000)
    assert info["top_n"] == 3
    assert info["pool_size"] == 3
    assert jnp.all(jnp.isfinite(f_path))
    assert selected.size >= 1
    assert not bool(jnp.any(jnp.isnan(info["epsilon"])))


def test_me_select_returns_vocab_indices():
    embeddings = jnp.eye(4)
    logits = jnp.array([-2.0, 4.0, 0.0, 1.0])
    selected, _, info = me_select(logits, embeddings, lambda_=0.9, top_n=2)
    for i in np.asarray(selected):
        assert 0 <= int(i) < 4
    cands = [int(i) for i in np.asarray(info["candidates"])]
    assert set(cands).issubset({0, 1, 2, 3})
    assert 1 in cands


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        greedy_select(jnp.array([0.5, 0.5]), jnp.eye(2), method="nope")
