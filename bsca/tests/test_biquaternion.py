"""Algebra tests for the biquaternion Hamilton product, norm, and inner product."""

import jax.numpy as jnp
import numpy as np

from bsca import bq_inner, bq_norm, hamilton
from bsca.biquaternion import one_like


def _basis(which, k=1, dtype=jnp.complex64):
    q = jnp.zeros((4, k), dtype=dtype)
    return q.at[which, :].set(jnp.ones((k,), dtype=dtype))


def _one(k=1, dtype=jnp.complex64):
    return _basis(0, k=k, dtype=dtype)


def test_i_j_k_square_to_minus_one():
    i, j, k = _basis(1), _basis(2), _basis(3)
    minus_one = -_one()
    for u in (i, j, k):
        np.testing.assert_allclose(np.asarray(hamilton(u, u)), np.asarray(minus_one), atol=1e-6)


def test_ij_k_and_ji_minus_k():
    i, j, k = _basis(1), _basis(2), _basis(3)
    np.testing.assert_allclose(np.asarray(hamilton(i, j)), np.asarray(k), atol=1e-6)
    np.testing.assert_allclose(np.asarray(hamilton(j, i)), np.asarray(-k), atol=1e-6)


def test_hamilton_bilinear_and_right_identity():
    rng = np.random.default_rng(0)
    q1 = jnp.asarray(rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3)), dtype=jnp.complex64)
    q2 = jnp.asarray(rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3)), dtype=jnp.complex64)
    q3 = jnp.asarray(rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3)), dtype=jnp.complex64)
    a = jnp.asarray(0.4 + 0.7j, dtype=jnp.complex64)
    b = jnp.asarray(-1.2 + 0.3j, dtype=jnp.complex64)
    left = hamilton(a * q1 + b * q2, q3)
    right = a * hamilton(q1, q3) + b * hamilton(q2, q3)
    np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=1e-5)
    ident = one_like(q1)
    np.testing.assert_allclose(np.asarray(hamilton(q1, ident)), np.asarray(q1), atol=1e-5)


def test_unit_quaternion_norm_is_one():
    q = jnp.asarray([[0.5], [0.5], [0.5], [0.5]], dtype=jnp.complex64)
    np.testing.assert_allclose(np.asarray(bq_norm(q)), 1.0, atol=1e-6)
    q1 = _one(k=1)
    np.testing.assert_allclose(np.asarray(bq_norm(q1)), 1.0, atol=1e-6)


def test_inner_product_real_and_equals_norm_squared():
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5)), dtype=jnp.complex64)
    inner = bq_inner(q, q)
    assert inner.shape == ()
    assert jnp.issubdtype(inner.dtype, jnp.floating)
    np.testing.assert_allclose(np.asarray(inner), np.asarray(bq_norm(q) ** 2), atol=1e-5)
    mixed = bq_inner(q, q * (0.3 + 0.4j))
    assert jnp.issubdtype(mixed.dtype, jnp.floating)
    assert np.isfinite(np.asarray(mixed))
