"""Identity, bound, and clustered reconstruction tests for CEDAR."""

import jax
import jax.numpy as jnp
import numpy as np

from cedar import (
    chunk_summaries,
    dense_attention,
    greedy_refine,
    hard_sparse_attention,
    hybrid_attention,
    reconstruction_bound,
    scale_tau,
)


def _clustered_problem(
    key,
    n_chunks=6,
    chunk_size=12,
    dim=8,
    noise=0.02,
    n_queries=None,
    scale=4.0,
):
    if n_queries is None:
        n_queries = min(5, n_chunks)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    centers_k = jax.random.normal(k1, (n_chunks, dim)) * scale
    centers_v = jax.random.normal(k2, (n_chunks, dim)) * scale
    jitter_k = noise * jax.random.normal(k3, (n_chunks, chunk_size, dim))
    jitter_v = noise * jax.random.normal(k4, (n_chunks, chunk_size, dim))
    k = (centers_k[:, None, :] + jitter_k).reshape(-1, dim)
    v = (centers_v[:, None, :] + jitter_v).reshape(-1, dim)
    q = centers_k[:n_queries] + 0.1 * jax.random.normal(
        jax.random.fold_in(key, 7), (n_queries, dim)
    )
    boundaries = jnp.arange(0, n_chunks * chunk_size + 1, chunk_size)
    return q, k, v, boundaries


def test_full_refinement_equals_dense():
    key = jax.random.PRNGKey(0)
    q, k, v, boundaries = _clustered_problem(key, noise=0.3)
    n_chunks = boundaries.shape[0] - 1
    refined = jnp.ones((q.shape[0], n_chunks), dtype=bool)
    tau = scale_tau(q.shape[-1], q.dtype)
    o_dense = dense_attention(q, k, v, tau=tau)
    o_cedar = hybrid_attention(q, k, v, boundaries, refined, tau=tau)
    np.testing.assert_allclose(np.asarray(o_cedar), np.asarray(o_dense), atol=1e-12, rtol=1e-12)


def test_zero_refinement_equals_length_corrected_chunk_attention():
    key = jax.random.PRNGKey(1)
    q, k, v, boundaries = _clustered_problem(key)
    n_chunks = boundaries.shape[0] - 1
    refined = jnp.zeros((q.shape[0], n_chunks), dtype=bool)
    tau = scale_tau(q.shape[-1], q.dtype)
    kbar, vbar, n, _, _ = chunk_summaries(k, v, boundaries)
    ell = (q @ kbar.T) / tau + jnp.log(n.astype(q.dtype))
    w = jnp.exp(ell - jax.nn.logsumexp(ell, axis=-1, keepdims=True))
    o_chunk = w @ vbar
    o_cedar = hybrid_attention(q, k, v, boundaries, refined, tau=tau)
    np.testing.assert_allclose(np.asarray(o_cedar), np.asarray(o_chunk), atol=1e-12, rtol=1e-12)


def test_no_double_counting():
    key = jax.random.PRNGKey(2)
    q, k, v, boundaries = _clustered_problem(key, n_chunks=4, chunk_size=8)
    n_chunks = boundaries.shape[0] - 1
    refined = jnp.zeros((q.shape[0], n_chunks), dtype=bool).at[:, 0].set(True)
    tau = scale_tau(q.shape[-1], q.dtype)
    o_correct = hybrid_attention(q, k, v, boundaries, refined, tau=tau)
    o_double = hybrid_attention(
        q, k, v, boundaries, refined, tau=tau, double_count=True
    )
    assert not np.allclose(np.asarray(o_correct), np.asarray(o_double), atol=1e-8)

    kbar, vbar, n, _, _ = chunk_summaries(k, v, boundaries)
    tok_logits = (q @ k.T) / tau
    chunk_of = jnp.repeat(jnp.arange(n_chunks), 8)
    tok_mask = refined[:, chunk_of]
    ninf = jnp.full((), -jnp.inf, dtype=q.dtype)
    tok_logits = jnp.where(tok_mask, tok_logits, ninf)
    ell = (q @ kbar.T) / tau + jnp.log(n.astype(q.dtype))
    sum_logits = jnp.where(~refined, ell, ninf)
    m = jnp.maximum(jnp.max(tok_logits, axis=-1), jnp.max(sum_logits, axis=-1))
    num = jnp.exp(tok_logits - m[:, None]) @ v + jnp.exp(sum_logits - m[:, None]) @ vbar
    den = jnp.exp(tok_logits - m[:, None]).sum(-1) + jnp.exp(sum_logits - m[:, None]).sum(-1)
    o_manual = num / den[:, None]
    np.testing.assert_allclose(np.asarray(o_correct), np.asarray(o_manual), atol=1e-12)


def test_proposition1_bound():
    key = jax.random.PRNGKey(3)
    q, k, v, boundaries = _clustered_problem(key, n_chunks=5, chunk_size=10, noise=0.15)
    n_chunks = boundaries.shape[0] - 1
    refined = jnp.zeros((q.shape[0], n_chunks), dtype=bool)
    refined = refined.at[:, :2].set(True)
    tau = scale_tau(q.shape[-1], q.dtype)
    o = dense_attention(q, k, v, tau=tau)
    hat = hybrid_attention(q, k, v, boundaries, refined, tau=tau)
    err = jnp.sqrt(jnp.sum((o - hat) ** 2, axis=-1))
    bound = reconstruction_bound(q, k, v, boundaries, refined, tau=tau)
    slack = 1e-9
    assert jnp.all(err <= bound + slack), (np.asarray(err), np.asarray(bound))
    assert jnp.all(bound >= 0)


def test_clustered_study_residual_beats_hard_drop():
    """Paper Sec. 6: same exact chunks via coarse similarity; CEDAR keeps summaries."""
    key = jax.random.PRNGKey(4)
    n_chunks, chunk_size, dim, n_queries, budget = 16, 8, 16, 16, 2
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    centers_k = jax.random.normal(k1, (n_chunks, dim))
    centers_v = jax.random.normal(k2, (n_chunks, dim))
    sigma = 0.05
    k = (centers_k[:, None, :] + sigma * jax.random.normal(k3, (n_chunks, chunk_size, dim))).reshape(-1, dim)
    v = (centers_v[:, None, :] + sigma * jax.random.normal(k4, (n_chunks, chunk_size, dim))).reshape(-1, dim)
    mix = jax.random.randint(k5, (n_queries, 3), 0, n_chunks)
    q = centers_k[mix].mean(axis=1)
    boundaries = jnp.arange(0, n_chunks * chunk_size + 1, chunk_size)
    tau = scale_tau(dim, q.dtype)
    kbar, _, n, _, _ = chunk_summaries(k, v, boundaries)
    sim = (q @ kbar.T) / tau
    refined = greedy_refine(sim, n, budget, unit="chunks")
    assert int(refined[0].sum()) == budget
    o = dense_attention(q, k, v, tau=tau)
    hat = hybrid_attention(q, k, v, boundaries, refined, tau=tau)
    hard = hard_sparse_attention(q, k, v, boundaries, refined, tau=tau)
    err_cedar = jnp.sqrt(jnp.sum((o - hat) ** 2, axis=-1)).mean()
    err_hard = jnp.sqrt(jnp.sum((o - hard) ** 2, axis=-1)).mean()
    reduction = 1.0 - err_cedar / err_hard
    assert err_hard > err_cedar
    assert reduction > 0.9, float(reduction)


def test_length_correction_mass_scales_with_n():
    dim = 4
    short, long = 3, 12
    vec = jnp.ones((dim,), dtype=jnp.float64)
    k_short = jnp.broadcast_to(vec, (short, dim))
    v_short = k_short
    k_long = jnp.broadcast_to(vec, (long, dim))
    v_long = k_long
    k = jnp.concatenate([k_short, k_long], axis=0)
    v = jnp.concatenate([v_short, v_long], axis=0)
    boundaries = jnp.array([0, short, short + long])
    q = jnp.ones((1, dim), dtype=jnp.float64)
    tau = scale_tau(dim, q.dtype)
    kbar, vbar, n, _, _ = chunk_summaries(k, v, boundaries)
    np.testing.assert_allclose(np.asarray(kbar[0]), np.asarray(kbar[1]), atol=1e-15)
    np.testing.assert_allclose(np.asarray(vbar[0]), np.asarray(vbar[1]), atol=1e-15)

    ell = (q @ kbar.T) / tau + jnp.log(n.astype(q.dtype))
    mass = jnp.exp(ell[0])
    ratio = mass[1] / mass[0]
    np.testing.assert_allclose(np.asarray(ratio), long / short, atol=1e-12)

    ell_nolog = (q @ kbar.T) / tau
    mass_nolog = jnp.exp(ell_nolog[0])
    np.testing.assert_allclose(np.asarray(mass_nolog[0]), np.asarray(mass_nolog[1]), atol=1e-12)


def test_bound_zero_when_fully_refined():
    key = jax.random.PRNGKey(5)
    q, k, v, boundaries = _clustered_problem(key, n_chunks=3, chunk_size=4)
    n_chunks = boundaries.shape[0] - 1
    refined = jnp.ones((q.shape[0], n_chunks), dtype=bool)
    bound = reconstruction_bound(q, k, v, boundaries, refined)
    np.testing.assert_allclose(np.asarray(bound), 0.0, atol=1e-15)
