"""Guess/slip affective modulation (eq. 26) and g_θ MLP (eq. 23)."""

import jax.numpy as jnp
import numpy as np

from aracd.affect import affect_mlp, guess_slip, modulate


def test_modulate_eq26_identity_and_endpoints():
    y_star = jnp.array([0.0, 0.3, 0.7, 1.0], dtype=jnp.float32)
    g0 = jnp.zeros_like(y_star)
    s0 = jnp.zeros_like(y_star)
    y_hat = modulate(y_star, g0, s0)
    np.testing.assert_allclose(np.asarray(y_hat), np.asarray(y_star), rtol=1e-6, atol=1e-6)

    g = jnp.array(0.2, dtype=jnp.float32)
    s = jnp.array(0.1, dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(modulate(jnp.array(0.0), g, s)), 0.2)
    np.testing.assert_allclose(np.asarray(modulate(jnp.array(1.0), g, s)), 1.0 - 0.1)

    y = jnp.array(0.4, dtype=jnp.float32)
    expected = (1.0 - 0.1) * 0.4 + 0.2 * (1.0 - 0.4)
    np.testing.assert_allclose(np.asarray(modulate(y, g, s)), expected, rtol=1e-6)


def test_guess_slip_heads():
    a_si = jnp.array([0.5, -0.25, 0.0], dtype=jnp.float32)
    w_g = jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32)
    w_s = jnp.array([0.0, 2.0, 0.0], dtype=jnp.float32)
    g, s = guess_slip(a_si, w_g, jnp.array(0.0), w_s, jnp.array(0.1))
    g_ref = 1.0 / (1.0 + np.exp(-(0.5 * 1.0 + 0.0)))
    s_ref = 1.0 / (1.0 + np.exp(-(-0.25 * 2.0 + 0.1)))
    np.testing.assert_allclose(np.asarray(g), g_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(s), s_ref, rtol=1e-6, atol=1e-6)


def test_affect_mlp_concat():
    a_s = jnp.array([0.1, -0.2], dtype=jnp.float32)
    d_kc = jnp.array([0.3, 0.4, 0.5], dtype=jnp.float32)
    w1 = jnp.ones((5, 3), dtype=jnp.float32) * 0.1
    b1 = jnp.zeros((3,), dtype=jnp.float32)
    w2 = jnp.ones((3, 2), dtype=jnp.float32) * 0.2
    b2 = jnp.array([0.01, -0.01], dtype=jnp.float32)
    a_si = affect_mlp(a_s, d_kc, w1, b1, w2, b2)
    z = np.concatenate([np.asarray(a_s), np.asarray(d_kc)], axis=-1)
    hidden = 1.0 / (1.0 + np.exp(-(z @ np.asarray(w1) + np.asarray(b1))))
    ref = hidden @ np.asarray(w2) + np.asarray(b2)
    np.testing.assert_allclose(np.asarray(a_si), ref, rtol=1e-5, atol=1e-6)
    assert a_si.shape == (2,)
