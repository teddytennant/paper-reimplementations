"""Eq. 6 allocation formulas."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_ga_amno.allocation import (
    allocate,
    direction_from_logits,
    displacements,
    importance_from_logits,
    importance_weights,
    project_domain,
)


def test_importance_weights_closed_form_and_sum():
    importance = jnp.asarray([0.1, 0.5, 0.9, 0.2], dtype=jnp.float32)
    i_min, s_i, eps = 0.05, 1.2, 1e-8
    w = importance_weights(importance, i_min=i_min, s_i=s_i, eps=eps)
    num = i_min + s_i * np.asarray(importance)
    expected = num / (num.sum() + eps)
    np.testing.assert_allclose(np.asarray(w), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(w)), float(num.sum() / (num.sum() + eps)), atol=1e-6)
    np.testing.assert_allclose(float(np.sum(w)), 1.0, atol=1e-5)


def test_displacements_elementwise():
    w = jnp.asarray([0.1, 0.4, 0.5], dtype=jnp.float32)
    o = jnp.asarray([[0.2, -0.3], [1.0, 0.0], [-0.5, 0.5]], dtype=jnp.float32)
    delta_max = 0.08
    dx = displacements(w, o, delta_max)
    expected = delta_max * np.asarray(w)[:, None] * np.asarray(o)
    np.testing.assert_allclose(np.asarray(dx), expected, rtol=1e-6, atol=1e-6)


def test_project_domain_clips_to_box():
    x = jnp.asarray([[-0.1, 0.5], [0.5, 1.2], [0.3, 0.4], [1.0, 0.0]], dtype=jnp.float32)
    xp = project_domain(x, omega_min=0.0, omega_max=1.0)
    np.testing.assert_allclose(
        np.asarray(xp),
        np.asarray([[0.0, 0.5], [0.5, 1.0], [0.3, 0.4], [1.0, 0.0]], dtype=np.float32),
        atol=1e-6,
    )


def test_allocate_matches_eq6_and_projects():
    x_r = jnp.asarray([[0.5, 0.5], [0.99, 0.01], [0.2, 0.2]], dtype=jnp.float32)
    importance = jnp.asarray([1.0, 0.25, 0.0], dtype=jnp.float32)
    direction = jnp.asarray([[1.0, -1.0], [1.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)
    i_min, s_i, eps, delta_max = 0.1, 2.0, 1e-8, 0.5
    x_a, w, dx = allocate(
        x_r,
        importance,
        direction,
        i_min=i_min,
        s_i=s_i,
        delta_max=delta_max,
        eps=eps,
        omega_min=0.0,
        omega_max=1.0,
    )
    num = i_min + s_i * np.asarray(importance)
    w_exp = num / (num.sum() + eps)
    dx_exp = delta_max * w_exp[:, None] * np.asarray(direction)
    np.testing.assert_allclose(np.asarray(w), w_exp, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(dx), dx_exp, rtol=1e-6, atol=1e-6)
    xa_exp = np.clip(np.asarray(x_r) + dx_exp, 0.0, 1.0)
    np.testing.assert_allclose(np.asarray(x_a), xa_exp, rtol=1e-6, atol=1e-6)
    assert np.all(np.asarray(x_a) >= 0.0) and np.all(np.asarray(x_a) <= 1.0)


def test_sigmoid_tanh_wrappers():
    logits = jnp.asarray([-10.0, 0.0, 10.0], dtype=jnp.float32)
    I = importance_from_logits(logits)
    np.testing.assert_allclose(np.asarray(I), 1.0 / (1.0 + np.exp(-np.asarray(logits))), atol=1e-5)
    o = direction_from_logits(logits)
    np.testing.assert_allclose(np.asarray(o), np.tanh(np.asarray(logits)), atol=1e-5)
