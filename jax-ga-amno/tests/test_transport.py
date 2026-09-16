"""Eqs. 8–10 transport, attention, residual, classic ablation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_ga_amno.transport import (
    apply_transport,
    attention_aggregate,
    residual_update,
    transport_matrix,
    transport_scale,
)


def _d_uvg(hidden=4, rank=2):
    d = jnp.asarray([0.5, 1.0, 1.5, 2.0], dtype=jnp.float32)
    u = jnp.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]],
        dtype=jnp.float32,
    )
    v = jnp.asarray(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [-1.0, 0.25]],
        dtype=jnp.float32,
    )
    gamma = jnp.asarray(0.4, dtype=jnp.float32)
    assert d.shape[0] == hidden and u.shape[1] == rank
    return d, u, v, gamma


def test_transport_shape_and_gamma_zero_is_diag():
    d, u, v, _ = _d_uvg()
    t0 = transport_matrix(d, u, v, jnp.asarray(0.0, dtype=jnp.float32))
    assert t0.shape == (d.shape[0], d.shape[0])
    np.testing.assert_allclose(np.asarray(t0), np.diag(np.asarray(d)), atol=1e-6)


def test_transport_zero_factors_is_diag_regardless_of_gamma():
    d, u, v, gamma = _d_uvg()
    z = jnp.zeros_like(u)
    t = transport_matrix(d, z, z, gamma)
    t2 = transport_matrix(d, z, v, jnp.asarray(1.0, dtype=jnp.float32))
    t3 = transport_matrix(d, u, z, jnp.asarray(1.0, dtype=jnp.float32))
    diag = np.diag(np.asarray(d))
    np.testing.assert_allclose(np.asarray(t), diag, atol=1e-6)
    np.testing.assert_allclose(np.asarray(t2), diag, atol=1e-6)
    np.testing.assert_allclose(np.asarray(t3), diag, atol=1e-6)


def test_transport_lowrank_scale_and_uv_doubling():
    d, u, v, gamma = _d_uvg()
    hidden, rank = int(d.shape[0]), int(u.shape[-1])
    t = transport_matrix(d, u, v, gamma)
    diag = np.diag(np.asarray(d))
    low = np.asarray(t) - diag
    uv = np.asarray(u) @ np.asarray(v).T
    expected = float(gamma) / np.sqrt(hidden * rank) * uv
    np.testing.assert_allclose(low, expected, rtol=1e-5, atol=1e-5)
    scale = float(transport_scale(hidden, rank))
    np.testing.assert_allclose(scale, 1.0 / np.sqrt(hidden * rank), atol=1e-6)
    t2 = transport_matrix(d, 2.0 * u, 2.0 * v, gamma)
    low2 = np.asarray(t2) - diag
    np.testing.assert_allclose(low2, 4.0 * low, rtol=1e-5, atol=1e-5)


def test_apply_transport_matches_matrix():
    d, u, v, gamma = _d_uvg()
    h = jnp.asarray([0.2, -0.1, 0.4, 0.7], dtype=jnp.float32)
    th = apply_transport(d, u, v, gamma, h)
    t = transport_matrix(d, u, v, gamma)
    np.testing.assert_allclose(np.asarray(th), np.asarray(t @ h), rtol=1e-5, atol=1e-5)


def test_aggregation_one_hot_identity_transport():
    h = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    neighbors = jnp.asarray([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
    h_j = h[neighbors]
    alpha = jnp.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    scores = jnp.asarray([[-50.0, 50.0], [50.0, -50.0], [-50.0, 50.0]], dtype=jnp.float32)
    m, a = attention_aggregate(h_j, scores)
    np.testing.assert_allclose(np.asarray(a), np.asarray(alpha), atol=1e-5)
    np.testing.assert_allclose(np.asarray(m[0]), np.asarray(h[2]), atol=1e-5)
    np.testing.assert_allclose(np.asarray(m[1]), np.asarray(h[0]), atol=1e-5)
    np.testing.assert_allclose(np.asarray(m[2]), np.asarray(h[1]), atol=1e-5)


def test_residual_identity_rho():
    h = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = jnp.asarray([[0.5, -1.0], [0.0, 0.25]], dtype=jnp.float32)
    h_new = residual_update(h, m, rho=None)
    np.testing.assert_allclose(np.asarray(h_new), np.asarray(h + m), atol=1e-6)
    h_new_id = residual_update(h, m, rho=lambda x: x)
    np.testing.assert_allclose(np.asarray(h_new_id), np.asarray(h + m), atol=1e-6)


def test_classic_ablation_differs_from_gauge_transport():
    hidden, rank, n, k = 4, 2, 3, 2
    d = jnp.ones((n, k, hidden), dtype=jnp.float32)
    u = jnp.ones((n, k, hidden, rank), dtype=jnp.float32)
    v = 0.3 * jnp.ones((n, k, hidden, rank), dtype=jnp.float32)
    gamma = jnp.ones((n, k), dtype=jnp.float32)
    h = jnp.arange(n * hidden, dtype=jnp.float32).reshape(n, hidden)
    neighbors = jnp.asarray([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
    h_j = h[neighbors]
    th_id = h_j
    th_g = apply_transport(d, u, v, gamma, h_j)
    scores = jnp.zeros((n, k), dtype=jnp.float32)
    m_id, _ = attention_aggregate(th_id, scores)
    m_g, _ = attention_aggregate(th_g, scores)
    assert not np.allclose(np.asarray(m_id), np.asarray(m_g), atol=1e-5)
    t = transport_matrix(d[0, 0], u[0, 0], v[0, 0], gamma[0, 0])
    assert not np.allclose(np.asarray(t), np.eye(hidden), atol=1e-5)
