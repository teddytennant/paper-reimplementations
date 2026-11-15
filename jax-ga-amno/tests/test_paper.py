"""Paper-equation checks: Eq. 6–10 plus identity geometry."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_ga_amno.allocation import allocate, displacements, importance_weights, project_domain
from jax_ga_amno.geometry import (
    edge_descriptors,
    four_connected_neighbors,
    local_jacobians,
    reference_grid,
)
from jax_ga_amno.model import GAAMNOConfig, ga_amno_forward, init_ga_amno
from jax_ga_amno.transport import apply_transport, attention_aggregate, residual_update, transport_matrix


def test_eq6_weights_displacements_projection():
    I = jnp.asarray([0.2, 0.8, 0.4], dtype=jnp.float32)
    i_min, s_i, eps = 0.05, 1.0, 1e-8
    w = importance_weights(I, i_min=i_min, s_i=s_i, eps=eps)
    num = i_min + s_i * np.asarray(I)
    np.testing.assert_allclose(np.asarray(w), num / (num.sum() + eps), atol=1e-6)
    np.testing.assert_allclose(float(np.sum(w)), 1.0, atol=1e-5)
    o = jnp.asarray([[0.3, -0.4], [0.9, 0.1], [-0.2, 0.7]], dtype=jnp.float32)
    dx = displacements(w, o, 0.08)
    np.testing.assert_allclose(
        np.asarray(dx), 0.08 * np.asarray(w)[:, None] * np.asarray(o), atol=1e-6
    )
    xa, _, _ = allocate(
        jnp.asarray([[0.0, 1.0], [0.5, 0.5], [1.1, -0.2]], dtype=jnp.float32),
        I,
        o,
        delta_max=0.08,
    )
    assert np.all(np.asarray(xa) >= 0.0) and np.all(np.asarray(xa) <= 1.0)
    np.testing.assert_allclose(
        np.asarray(project_domain(jnp.asarray([[-1.0, 2.0]]))),
        np.asarray([[0.0, 1.0]]),
        atol=1e-6,
    )


def test_eq8_transport_identities():
    d = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    u = jnp.asarray([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    v = jnp.asarray([[0.5, 0.0], [1.0, 1.0], [0.0, 2.0]], dtype=jnp.float32)
    gamma = jnp.asarray(0.7, dtype=jnp.float32)
    c_h, r = 3, 2
    t = transport_matrix(d, u, v, gamma)
    assert t.shape == (c_h, c_h)
    low = np.asarray(t) - np.diag(np.asarray(d))
    np.testing.assert_allclose(
        low,
        float(gamma) / np.sqrt(c_h * r) * (np.asarray(u) @ np.asarray(v).T),
        rtol=1e-5,
        atol=1e-5,
    )
    t_g0 = transport_matrix(d, u, v, jnp.asarray(0.0, dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(t_g0), np.diag(np.asarray(d)), atol=1e-6)
    t_uv0 = transport_matrix(d, jnp.zeros_like(u), jnp.zeros_like(v), gamma)
    np.testing.assert_allclose(np.asarray(t_uv0), np.diag(np.asarray(d)), atol=1e-6)
    t_dbl = transport_matrix(d, 2 * u, 2 * v, gamma)
    np.testing.assert_allclose(
        np.asarray(t_dbl) - np.diag(np.asarray(d)), 4.0 * low, rtol=1e-5, atol=1e-5
    )


def test_eq9_eq10_mix_and_residual():
    h = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)
    neighbors = jnp.asarray([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
    scores = jnp.asarray([[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]], dtype=jnp.float32)
    m, alpha = attention_aggregate(h[neighbors], scores)
    np.testing.assert_allclose(np.asarray(m[0]), np.asarray(h[1]), atol=1e-5)
    np.testing.assert_allclose(np.asarray(alpha[0]), np.asarray([1.0, 0.0]), atol=1e-5)
    h_new = residual_update(h, m, rho=None)
    np.testing.assert_allclose(np.asarray(h_new), np.asarray(h + m), atol=1e-6)


def test_identity_geometry_four_neighbors_jacobian():
    height = width = 8
    x_r = reference_grid(height, width)
    x_a = x_r
    neighbors = four_connected_neighbors(height, width)
    i = 3 * width + 3
    rel = np.asarray(x_a[neighbors[i]] - x_a[i])
    # 4-neighbors are axis-aligned: exactly one coordinate differs.
    np.testing.assert_array_equal(np.sum(np.abs(rel) > 1e-6, axis=-1), np.array([1, 1, 1, 1]))
    # left/right along x, down/up along y
    np.testing.assert_allclose(rel[0, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(rel[1, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(rel[2, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(rel[3, 0], 0.0, atol=1e-6)
    assert rel[0, 0] < 0 and rel[1, 0] > 0
    assert rel[2, 1] < 0 and rel[3, 1] > 0
    J = local_jacobians(x_a, x_r, neighbors)
    np.testing.assert_allclose(np.asarray(J[i]), np.eye(2), atol=1e-4, rtol=1e-4)
    interior = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            interior.append(y * width + x)
    J_int = np.asarray(J[jnp.asarray(interior)])
    for ji in J_int:
        np.testing.assert_allclose(ji, np.eye(2), atol=1e-4, rtol=1e-4)


def test_eq7_edge_descriptor_concat():
    height = width = 4
    x_r = reference_grid(height, width)
    x_a = x_r + 0.01
    neighbors = four_connected_neighbors(height, width)
    J = local_jacobians(x_a, x_r, neighbors)
    p = jnp.arange(height * width * 3, dtype=jnp.float32).reshape(height * width, 3)
    e = edge_descriptors(x_a, J, p, neighbors)
    n, k = neighbors.shape
    assert e.shape == (n, k, 2 + 4 + 4 + 3 + 3)
    rel = np.asarray(x_a[neighbors] - x_a[:, None, :])
    np.testing.assert_allclose(np.asarray(e[..., :2]), rel, atol=1e-5)
    j_j = np.asarray(J[neighbors]).reshape(n, k, 4)
    j_i = np.broadcast_to(np.asarray(J).reshape(n, 1, 4), (n, k, 4))
    np.testing.assert_allclose(np.asarray(e[..., 2:6]), j_j, atol=1e-5)
    np.testing.assert_allclose(np.asarray(e[..., 6:10]), j_i, atol=1e-5)
    np.testing.assert_allclose(np.asarray(e[..., 10:13]), np.asarray(p[neighbors]), atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(e[..., 13:16]),
        np.broadcast_to(np.asarray(p)[:, None, :], (n, k, 3)),
        atol=1e-5,
    )


def test_classic_vs_gauge_forward_differ():
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (8, 8))
    cfg_g = GAAMNOConfig(height=8, width=8, hidden=6, rank=2, n_layers=2, skip_transport=False)
    cfg_c = cfg_g._replace(skip_transport=True)
    params = init_ga_amno(key, cfg_g)
    layers = []
    for layer in params["layers"]:
        layer = dict(layer)
        layer["W_U"] = layer["W_U"] * 20.0
        layer["W_V"] = layer["W_V"] * 20.0
        layer["b_gamma"] = jnp.asarray([2.0])
        layers.append(layer)
    params = {**params, "layers": layers}
    out_g = ga_amno_forward(params, field, cfg_g)
    out_c = ga_amno_forward(params, field, cfg_c)
    assert not np.allclose(np.asarray(out_g["m"]), np.asarray(out_c["m"]), atol=1e-5)
    assert not np.allclose(np.asarray(out_g["h"]), np.asarray(out_c["h"]), atol=1e-5)


def test_apply_matches_materialised_T_on_neighbors():
    d = jnp.ones((2, 3), dtype=jnp.float32)
    u = jnp.ones((2, 3, 1), dtype=jnp.float32)
    v = 0.2 * jnp.ones((2, 3, 1), dtype=jnp.float32)
    gamma = jnp.asarray([0.5, 1.0], dtype=jnp.float32)
    h = jnp.asarray([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]], dtype=jnp.float32)
    th = apply_transport(d, u, v, gamma, h)
    for i in range(2):
        t = transport_matrix(d[i], u[i], v[i], gamma[i])
        np.testing.assert_allclose(np.asarray(th[i]), np.asarray(t @ h[i]), atol=1e-5)
