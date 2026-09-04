"""Lemma B.1 and the discounted (N+1)-th order predictor of eq. (22)."""

import jax
import jax.numpy as jnp
import numpy as np

from hood.predictors import (
    discount_factor,
    discounted_difference,
    first_order_predictor,
    hood_predictor,
    predictor_sequence,
)


def test_lemma_b1_identity():
    """eq. (B.5): e = sigma^{N+1} Delta^{N+1} g, with e = g - m."""
    key = jax.random.key(0)
    for n in (1, 2, 3):
        key, sub = jax.random.split(key)
        g = jax.random.normal(sub, (30, n, 4))
        m = predictor_sequence(g, n)
        err = g - m
        rhs = discounted_difference(g, n)
        np.testing.assert_allclose(err, rhs, rtol=1e-8, atol=1e-10)


def test_zero_prehistory_first_prediction_is_zero():
    g = jax.random.normal(jax.random.key(1), (12, 2, 3))
    m = predictor_sequence(g, 2)
    np.testing.assert_allclose(np.asarray(m[0]), 0.0, atol=1e-15)


def test_discount_factor():
    assert discount_factor(1) == 1.0 / 2.0
    assert discount_factor(2) == 2.0 / 3.0
    assert discount_factor(3) == 3.0 / 4.0


def test_first_order_predictor_returns_g_buffer_zero():
    g_buf = jax.random.normal(jax.random.key(2), (3, 2, 4))
    m_buf = jax.random.normal(jax.random.key(3), (3, 2, 4))
    out = first_order_predictor(g_buf, m_buf, 2)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(g_buf[0]))


def test_hood_predictor_zero_buffers():
    g_buf = jnp.zeros((3, 2, 4))
    m_buf = jnp.zeros((3, 2, 4))
    out = hood_predictor(g_buf, m_buf, 2)
    np.testing.assert_allclose(np.asarray(out), 0.0)
