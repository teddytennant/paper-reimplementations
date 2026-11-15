"""Lock displayed BridgeMem equations (arXiv:2609.29268) on CPU."""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from bridgemem.constants import EPS, N_BINS, TAU
from bridgemem.kernel import (
    A_ct,
    A_nn,
    attention_pool,
    event_embedding,
    final_log_score,
    g_gate,
    init_params,
    layer_norm,
    log_scores,
    mix_weight_count,
    mixed_residual,
    mlp_head,
    neural_log_probs,
    recency_bin,
    residual_from_tokens,
    rho_B,
)
from bridgemem.memory import recency_bin_np, retrieve_pair, stack_candidates


def _params_hand():
    """Tiny embeddings so eq. 1–2 can be evaluated by hand."""
    dim = 2
    n_rel = 2
    hidden = 3
    n_dir = 2 * n_rel
    return {
        "e_r": jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        "e_d": jnp.array([[0.5, 0.0], [0.0, 0.5]], dtype=jnp.float32),
        "e_b": jnp.array(
            [[0.1, 0.0], [0.0, 0.1], [0.2, 0.2], [0.0, 0.0], [0.3, -0.1]],
            dtype=jnp.float32,
        ),
        "w": jnp.array([0.4, -0.2], dtype=jnp.float32),
        "ln_scale": jnp.ones((dim,), dtype=jnp.float32),
        "ln_bias": jnp.zeros((dim,), dtype=jnp.float32),
        "W1": jnp.array([[0.2, -0.1, 0.3], [0.0, 0.4, -0.2]], dtype=jnp.float32),
        "b1": jnp.zeros((hidden,), dtype=jnp.float32),
        "W2": jnp.array(
            [[0.1, 0.0, -0.1, 0.2], [0.0, 0.1, 0.0, -0.1], [0.2, -0.2, 0.1, 0.0]],
            dtype=jnp.float32,
        ),
        "b2": jnp.zeros((n_dir,), dtype=jnp.float32),
    }


def test_recency_bins_match_displayed_edges():
    # edges {0, 1, 4, 16, 64} -> 5 bins (0,1], (1,4], (4,16], (16,64], (64, inf)
    lags = jnp.array([1, 2, 4, 5, 16, 17, 64, 65, 100])
    bins = recency_bin(lags)
    assert bins.tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert int(recency_bin(jnp.array(1))) == 0
    assert N_BINS == 5
    np.testing.assert_array_equal(np.array(bins), recency_bin_np(np.array(lags)))


def test_eq1_event_embedding_is_sum_of_three_tables():
    p = _params_hand()
    rel = jnp.array([1, 0])
    d = jnp.array([0, 1])
    b = jnp.array([2, 0])
    z = event_embedding(p, rel, d, b)
    z0 = p["e_r"][1] + p["e_d"][0] + p["e_b"][2]
    z1 = p["e_r"][0] + p["e_d"][1] + p["e_b"][0]
    np.testing.assert_allclose(z[0], z0, rtol=1e-6)
    np.testing.assert_allclose(z[1], z1, rtol=1e-6)


def test_eq1_attention_is_softmax_of_w_tanh_z():
    p = _params_hand()
    z = jnp.array([[0.3, -0.2], [1.0, 0.5], [0.0, 0.0]], dtype=jnp.float32)
    mask = jnp.array([True, True, False])
    alpha, m = attention_pool(p, z, mask)
    scores_np = np.array(jnp.tanh(z) @ p["w"])
    scores_np[2] = -1.0e9
    alpha_np = np.exp(scores_np - scores_np.max())
    alpha_np = alpha_np / alpha_np.sum()
    np.testing.assert_allclose(alpha, alpha_np, rtol=1e-5)
    m_np = alpha_np[:, None] * np.array(z)
    m_np = m_np.sum(axis=0)
    np.testing.assert_allclose(m, m_np, rtol=1e-5)


def test_attention_permutation_consistent():
    p = _params_hand()
    rel = jnp.array([0, 1, 0, 1])
    d = jnp.array([0, 1, 1, 0])
    b = jnp.array([0, 1, 2, 3])
    mask = jnp.array([True, True, True, False])
    z = event_embedding(p, rel, d, b)
    _, m = attention_pool(p, z, mask)
    perm = jnp.array([2, 0, 1, 3])
    z_p = z[perm]
    mask_p = mask[perm]
    _, m_p = attention_pool(p, z_p, mask_p)
    np.testing.assert_allclose(m, m_p, rtol=1e-5, atol=1e-6)

    log_pi = jnp.log(jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32))
    p_ct = jnp.ones((2, 2, 5, 4), dtype=jnp.float32) / 4.0
    token_n = jnp.ones((2, 2, 5), dtype=jnp.float32)
    A = residual_from_tokens(p, rel, d, b, mask, p_ct, token_n, log_pi, 1)
    A_p = residual_from_tokens(
        p, rel[perm], d[perm], b[perm], mask[perm], p_ct, token_n, log_pi, 1
    )
    np.testing.assert_allclose(A, A_p, rtol=1e-5, atol=1e-6)


def test_eq2_neural_reader_adds_log_pi_then_subtracts():
    p = _params_hand()
    m = jnp.array([0.5, -0.25], dtype=jnp.float32)
    pi = jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)
    log_pi = jnp.log(pi)
    log_p = neural_log_probs(p, m, log_pi)
    h = layer_norm(m, p["ln_scale"], p["ln_bias"])
    h = jax.nn.relu(h @ p["W1"] + p["b1"])
    logits = h @ p["W2"] + p["b2"] + log_pi
    log_p_np = logits - jax.nn.logsumexp(logits)
    np.testing.assert_allclose(log_p, log_p_np, rtol=1e-5)
    a = A_nn(log_p, log_pi, 1)
    assert float(a) == pytest.approx(float(log_p[1] - log_pi[1]), rel=1e-6)
    # unused head output still exists; formula is not MLP(m) without LN or π
    raw = mlp_head(p, m)
    assert not np.allclose(raw, log_p)


def test_eq3_count_reader_is_log_sum_minus_log_pi():
    mask = jnp.array([True, True, False])
    p_ctx = jnp.array([0.2, 0.5, 0.9], dtype=jnp.float32)
    log_pi_r = jnp.log(jnp.float32(0.25))
    a = A_ct(p_ctx, log_pi_r, mask)
    expected = np.log(0.2 + 0.5) - np.log(0.25)
    assert float(a) == pytest.approx(expected, rel=1e-6)
    # padded 0.9 must not enter the sum
    assert float(a) != pytest.approx(np.log(0.2 + 0.5 + 0.9) - np.log(0.25), rel=1e-4)


def test_eq4_mix_is_log_of_weighted_exponentials_tau_100():
    A_count = jnp.float32(1.2)
    A_neural = jnp.float32(-0.4)
    n_B = jnp.float32(50.0)
    rho = rho_B(n_B, TAU)
    assert float(rho) == pytest.approx(100.0 / (50.0 + 100.0), rel=1e-6)
    A = mixed_residual(A_count, A_neural, n_B, TAU)
    expected = np.log((1.0 - 100.0 / 150.0) * np.exp(1.2) + (100.0 / 150.0) * np.exp(-0.4))
    assert float(A) == pytest.approx(expected, rel=1e-5)
    # not an interpolation of the log-ratios themselves
    linear = (1.0 - 100.0 / 150.0) * 1.2 + (100.0 / 150.0) * (-0.4)
    assert abs(float(A) - linear) > 1e-3
    # n_B = 0 => all neural
    assert float(mixed_residual(A_count, A_neural, 0.0, TAU)) == pytest.approx(-0.4, rel=1e-5)


def test_empirical_bayes_trusts_counts_when_abundant():
    w_abundant = float(mix_weight_count(1000.0, TAU))
    w_sparse = float(mix_weight_count(2.0, TAU))
    assert w_abundant > w_sparse
    assert w_abundant == pytest.approx(1000.0 / (1000.0 + 100.0), rel=1e-6)
    assert w_sparse == pytest.approx(2.0 / (2.0 + 100.0), rel=1e-6)


def test_eq5_gate_and_log_score():
    u = jnp.array([0.0, 0.5, 1.0, 0.25], dtype=jnp.float32)
    g = g_gate(u)
    np.testing.assert_allclose(g, 4.0 * u * (1.0 - u), rtol=1e-6)
    assert float(g[0]) == pytest.approx(0.0, abs=1e-7)
    assert float(g[1]) == pytest.approx(1.0, rel=1e-6)
    assert float(g[2]) == pytest.approx(0.0, abs=1e-7)
    p0 = jnp.float32(0.2)
    A = jnp.float32(1.5)
    lam = 2.0
    S = final_log_score(p0, A, jnp.float32(0.5), lam=lam)
    assert float(S) == pytest.approx(np.log(0.2) + 2.0 * 1.0 * 1.5, rel=1e-6)
    S_floor = final_log_score(jnp.float32(0.0), A, jnp.float32(0.5), lam=0.0)
    assert float(S_floor) == pytest.approx(np.log(EPS), rel=1e-6)


def test_empty_dyadic_history_residual_is_zero():
    p = _params_hand()
    rel = jnp.zeros((4,), dtype=jnp.int32)
    d = jnp.zeros((4,), dtype=jnp.int32)
    b = jnp.zeros((4,), dtype=jnp.int32)
    mask = jnp.zeros((4,), dtype=bool)
    log_pi = jnp.log(jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32))
    p_ct = jnp.ones((2, 2, 5, 4), dtype=jnp.float32) / 4.0
    token_n = jnp.ones((2, 2, 5), dtype=jnp.float32) * 50.0
    A = residual_from_tokens(p, rel, d, b, mask, p_ct, token_n, log_pi, 0)
    assert float(A) == pytest.approx(0.0, abs=1e-7)
    p0 = jnp.float32(0.3)
    S, A2 = log_scores(
        p, rel, d, b, mask, p_ct, token_n, log_pi, 0, p0, jnp.float32(0.5)
    )
    assert float(A2) == pytest.approx(0.0, abs=1e-7)
    assert float(S) == pytest.approx(np.log(0.3), rel=1e-6)


def test_high_backbone_confidence_leaves_scores_unchanged():
    p = _params_hand()
    rel = jnp.array([0, 1, 0, 0], dtype=jnp.int32)
    d = jnp.array([0, 0, 1, 0], dtype=jnp.int32)
    b = jnp.array([1, 2, 0, 0], dtype=jnp.int32)
    mask = jnp.array([True, True, True, False])
    log_pi = jnp.log(jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32))
    p_ct = jnp.ones((2, 2, 5, 4), dtype=jnp.float32) / 4.0
    p_ct = p_ct.at[0, 0, 1, 1].set(0.8)
    token_n = jnp.ones((2, 2, 5), dtype=jnp.float32) * 10.0
    p0 = jnp.float32(0.15)
    S_conf, A = log_scores(
        p, rel, d, b, mask, p_ct, token_n, log_pi, 1, p0, jnp.float32(1.0), lam=1.0
    )
    assert abs(float(A)) > 1e-6
    assert float(S_conf) == pytest.approx(np.log(0.15), rel=1e-5, abs=1e-6)
    S0, _ = log_scores(
        p, rel, d, b, mask, p_ct, token_n, log_pi, 1, p0, jnp.float32(0.0), lam=1.0
    )
    assert float(S0) == pytest.approx(np.log(0.15), rel=1e-5, abs=1e-6)


def test_events_at_or_after_t_never_used():
    t = 10
    events = np.array(
        [
            [0, 0, 1, 3],
            [0, 1, 1, 7],
            [0, 0, 1, 10],
            [1, 1, 0, 11],
            [0, 1, 1, 12],
        ],
        dtype=np.int32,
    )
    tok = retrieve_pair(events, actor=0, candidate=1, t=t)
    used = tok["time"][tok["mask"]]
    assert np.all(used < t)
    assert 10 not in used
    assert 11 not in used
    assert 12 not in used
    assert set(used.tolist()) == {3, 7}

    events_causal = events[events[:, 3] < t]
    tok2 = retrieve_pair(events_causal, actor=0, candidate=1, t=t)
    np.testing.assert_array_equal(tok["relation"][tok["mask"]], tok2["relation"][tok2["mask"]])
    np.testing.assert_array_equal(tok["time"][tok["mask"]], tok2["time"][tok2["mask"]])


def test_window_drops_events_older_than_65():
    t = 100
    events = np.array(
        [
            [0, 0, 1, 20],
            [0, 1, 1, 40],
            [0, 0, 1, 90],
        ],
        dtype=np.int32,
    )
    tok = retrieve_pair(events, actor=0, candidate=1, t=t)
    used = set(tok["time"][tok["mask"]].tolist())
    assert 20 not in used
    assert 90 in used


def test_residual_changes_ranking_with_supporting_transition():
    p = init_params(jax.random.PRNGKey(0), n_rel=2, dim=8, hidden=8)
    n_ent = 3
    n_dir = 4
    log_pi = jnp.log(jnp.ones((n_dir,), dtype=jnp.float32) / n_dir)
    p_ct = jnp.ones((2, 2, 5, n_dir), dtype=jnp.float32) / n_dir
    # token (d=0, r=0, bin of lag 3) strongly predicts query relation 1
    lag = 3
    b = int(recency_bin(jnp.array(lag)))
    p_ct = p_ct.at[0, 0, b, 1].set(0.85)
    token_n = jnp.ones((2, 2, 5), dtype=jnp.float32)
    token_n = token_n.at[0, 0, b].set(200.0)

    rel = jnp.zeros((n_ent, 8), dtype=jnp.int32)
    d = jnp.zeros((n_ent, 8), dtype=jnp.int32)
    bins = jnp.zeros((n_ent, 8), dtype=jnp.int32)
    mask = jnp.zeros((n_ent, 8), dtype=bool)
    # candidate 1 has a supporting 0->1 event
    rel = rel.at[1, 0].set(0)
    d = d.at[1, 0].set(0)
    bins = bins.at[1, 0].set(b)
    mask = mask.at[1, 0].set(True)

    p0 = jnp.ones((n_ent,), dtype=jnp.float32) / n_ent
    u_q = jnp.float32(0.5)
    S, A = log_scores(p, rel, d, bins, mask, p_ct, token_n, log_pi, 1, p0, u_q)
    backbone = jnp.log(p0)
    np.testing.assert_allclose(backbone[0], backbone[1], rtol=1e-6)
    assert float(A[1]) > float(A[0])
    assert float(A[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(S[1]) > float(S[0])
    assert int(np.argmax(np.array(S))) == 1
