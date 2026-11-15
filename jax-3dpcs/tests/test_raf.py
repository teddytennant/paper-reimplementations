"""Tensor RAF gates (eqs. 9-11), no backbone."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_3dpcs.raf import (
    attention_weights,
    fuse_features,
    residual_gate,
    residual_regression,
)


def test_gate_zero_leaves_coarse_estimate_unchanged():
    y_c = jnp.array([1.25, -0.5, 4.0], dtype=jnp.float32)
    delta = jnp.array([10.0, 20.0, -3.0], dtype=jnp.float32)
    out = residual_regression(y_c, delta, gate=0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(y_c), atol=1e-6)


def test_gate_one_applies_residual_as_written():
    y_c = jnp.array([1.25, -0.5, 4.0], dtype=jnp.float32)
    delta = jnp.array([10.0, 20.0, -3.0], dtype=jnp.float32)
    out = residual_regression(y_c, delta, gate=1.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(y_c + delta), atol=1e-6)


def test_default_gate_matches_eq11():
    y_c = jnp.array([0.1, 0.2], dtype=jnp.float32)
    delta = jnp.array([0.3, -0.4], dtype=jnp.float32)
    out = residual_regression(y_c, delta)
    np.testing.assert_allclose(np.asarray(out), np.asarray(y_c + delta), atol=1e-6)


def test_feature_gate_zero_leaves_features_unchanged():
    x = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
    w = jnp.zeros_like(x)
    out = residual_gate(x, w)
    np.testing.assert_allclose(np.asarray(out), np.asarray(x), atol=1e-6)


def test_feature_gate_one_doubles_features():
    x = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
    w = jnp.ones_like(x)
    out = residual_gate(x, w)
    np.testing.assert_allclose(np.asarray(out), np.asarray(x + x), atol=1e-6)


def test_attention_weights_sigmoid():
    logits = jnp.array([0.0, 1e6, -1e6], dtype=jnp.float32)
    w = attention_weights(logits)
    np.testing.assert_allclose(float(w[0]), 0.5, atol=1e-5)
    np.testing.assert_allclose(float(w[1]), 1.0, atol=1e-5)
    np.testing.assert_allclose(float(w[2]), 0.0, atol=1e-5)


def test_fuse_features_concat_after_gates():
    x_c = jnp.array([1.0, 2.0], dtype=jnp.float32)
    x_l = jnp.array([3.0, 4.0], dtype=jnp.float32)
    w_c = jnp.array([0.0, 1.0], dtype=jnp.float32)
    w_l = jnp.array([1.0, 0.0], dtype=jnp.float32)
    fused = fuse_features(x_c, x_l, w_c, w_l)
    expected = np.array([1.0, 4.0, 6.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(fused), expected, atol=1e-6)
