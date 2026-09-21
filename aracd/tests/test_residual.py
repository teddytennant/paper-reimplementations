"""Ability residual, stu-item residual, logit correction (eqs. 7, 10–12)."""

import jax.numpy as jnp
import numpy as np

from aracd.ops import logit, sigmoid
from aracd.residual import (
    GAMMA_SI_DEFAULT,
    ability_correct,
    ability_residual,
    stu_item_residual,
)


def test_logit_residual_sigmoid_eq7():
    y_base = jnp.array(0.7, dtype=jnp.float32)
    r = jnp.array(0.3, dtype=jnp.float32)
    y_star = ability_correct(y_base, r)
    z = float(logit(y_base)) + 0.3
    expected = 1.0 / (1.0 + np.exp(-z))
    np.testing.assert_allclose(np.asarray(y_star), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(y_star), np.asarray(sigmoid(logit(y_base) + r)))


def test_stu_item_residual_eq12():
    b_s = jnp.array(0.2, dtype=jnp.float32)
    b_i = jnp.array(-0.1, dtype=jnp.float32)
    u_s = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    v_i = jnp.array([0.5, -0.5, 0.25, 0.0], dtype=jnp.float32)
    gamma = GAMMA_SI_DEFAULT
    d = 4
    dot = float(np.dot(np.asarray(u_s), np.asarray(v_i)))
    expected = 0.2 + (-0.1) + gamma * dot / np.sqrt(d)
    got = stu_item_residual(b_s, b_i, u_s, v_i, gamma_si=gamma)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-6, atol=1e-6)


def test_stu_item_zero_bias_is_only_interaction():
    u_s = jnp.array([0.4, -0.2], dtype=jnp.float32)
    v_i = jnp.array([1.0, 0.5], dtype=jnp.float32)
    r = stu_item_residual(0.0, 0.0, u_s, v_i, gamma_si=1.5)
    d = 2
    expected = 1.5 * float(np.dot(np.asarray(u_s), np.asarray(v_i))) / np.sqrt(d)
    np.testing.assert_allclose(np.asarray(r), expected, rtol=1e-6, atol=1e-6)


def test_ability_residual_eq10():
    r_si = jnp.array(0.4, dtype=jnp.float32)
    r_c = jnp.array(-0.2, dtype=jnp.float32)
    got = ability_residual(r_si, r_c, w_si=jnp.array(1.5), w_c=jnp.array(0.5))
    np.testing.assert_allclose(np.asarray(got), 1.5 * 0.4 + 0.5 * (-0.2))


def test_zero_residual_leaves_y_star_equal_y_base():
    y_base = jnp.array([0.2, 0.5, 0.8], dtype=jnp.float32)
    r = jnp.zeros_like(y_base)
    y_star = ability_correct(y_base, r)
    np.testing.assert_allclose(np.asarray(y_star), np.asarray(y_base), rtol=1e-5, atol=1e-6)
