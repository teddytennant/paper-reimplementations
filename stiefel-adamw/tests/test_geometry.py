"""Projection and retractions on St(n, r) (Appendix A.1–A.2)."""

import jax
import jax.numpy as jnp
import pytest

from stiefel_adamw.geometry import (
    cayley_retract,
    project,
    project_a1,
    qr_retract,
    random_stiefel,
    skew,
    sym,
)

jax.config.update("jax_enable_x64", True)


def _tangent(x, z):
    return project(x, z)


def test_projection_is_tangent_and_formulas_agree():
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    n, r = 12, 5
    x = random_stiefel(k1, n, r, dtype=jnp.float64)
    z = jax.random.normal(k2, (n, r), dtype=jnp.float64)
    p_compact = project(x, z)
    p_a1 = project_a1(x, z)
    assert jnp.allclose(p_compact, p_a1, atol=1e-10)
    xtp = x.T @ p_compact
    assert jnp.allclose(xtp + xtp.T, 0.0, atol=1e-5)
    assert jnp.allclose(xtp, skew(xtp), atol=1e-5)


def test_projection_identity_on_tangent_kills_normal():
    key = jax.random.key(1)
    k1, k2, k3 = jax.random.split(key, 3)
    n, r = 10, 4
    x = random_stiefel(k1, n, r, dtype=jnp.float64)
    z = jax.random.normal(k2, (n, r), dtype=jnp.float64)
    xi = _tangent(x, z)
    assert jnp.allclose(project(x, xi), xi, atol=1e-8)
    assert jnp.allclose(project_a1(x, xi), xi, atol=1e-8)

    s = sym(jax.random.normal(k3, (r, r), dtype=jnp.float64))
    normal = x @ s
    assert jnp.allclose(project(x, normal), 0.0, atol=1e-8)
    assert jnp.allclose(project(x, xi + normal), xi, atol=1e-8)


def test_qr_retraction_stays_on_stiefel_and_fixes_zero():
    key = jax.random.key(2)
    k1, k2 = jax.random.split(key)
    n, r = 11, 4
    x = random_stiefel(k1, n, r, dtype=jnp.float64)
    z = jax.random.normal(k2, (n, r), dtype=jnp.float64)
    xi = _tangent(x, 0.3 * z)
    y = qr_retract(x, xi)
    assert jnp.allclose(y.T @ y, jnp.eye(r), atol=1e-8)
    y0 = qr_retract(x, jnp.zeros_like(x))
    assert jnp.allclose(y0, x, atol=1e-8)


def test_cayley_fixed_point_on_manifold_zero_and_first_order():
    key = jax.random.key(3)
    k1, k2 = jax.random.split(key)
    n, r = 10, 4
    x = random_stiefel(k1, n, r, dtype=jnp.float64)
    z = jax.random.normal(k2, (n, r), dtype=jnp.float64)
    xi = _tangent(x, 0.05 * z)

    y = cayley_retract(x, xi, alpha=1.0, n_iters=15)
    assert jnp.allclose(y.T @ y, jnp.eye(r), atol=1e-5)

    y0 = cayley_retract(x, jnp.zeros_like(x), n_iters=4)
    assert jnp.allclose(y0, x, atol=1e-10)

    eps = 1e-4
    y_eps = cayley_retract(x, eps * xi, alpha=1.0, n_iters=8)
    fd = (y_eps - x) / eps
    rel = jnp.linalg.norm(fd - xi) / (jnp.linalg.norm(xi) + 1e-15)
    assert rel < 0.05
