"""NCDM interaction and backbone (eqs. 8–9)."""

import jax.numpy as jnp
import numpy as np

from aracd.ncdm import ncdm_base, ncdm_interaction, ncdm_mlp
from aracd.ops import sigmoid


def test_ncdm_interaction_matches_hand_and_q_mask():
    e_s = jnp.array([0.8, 0.3, 0.6], dtype=jnp.float32)
    d_kc = jnp.array([0.4, 0.5, 0.2], dtype=jnp.float32)
    d_disc = jnp.array([1.2, 0.9, 1.1], dtype=jnp.float32)
    q_i = jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32)

    x = ncdm_interaction(e_s, d_disc, d_kc, q_i)
    expected = np.array(d_disc) * (np.array(e_s) - np.array(d_kc)) * np.array(q_i)
    np.testing.assert_allclose(np.asarray(x), expected, rtol=1e-6, atol=1e-6)

    assert float(x[1]) == 0.0
    np.testing.assert_allclose(np.asarray(x[0]), np.float32(1.2) * (np.float32(0.8) - np.float32(0.4)), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(x[2]), np.float32(1.1) * (np.float32(0.6) - np.float32(0.2)), rtol=1e-6, atol=1e-6)


def test_ncdm_base_sigmoid_mlp():
    e_s = jnp.array([0.7, 0.2], dtype=jnp.float32)
    d_kc = jnp.array([0.3, 0.4], dtype=jnp.float32)
    d_disc = jnp.array([0.9, 1.1], dtype=jnp.float32)
    q_i = jnp.array([1.0, 1.0], dtype=jnp.float32)
    w1 = jnp.array([[0.2, -0.1], [0.3, 0.4]], dtype=jnp.float32)
    b1 = jnp.array([0.05, -0.02], dtype=jnp.float32)
    w2 = jnp.array([0.4, -0.3], dtype=jnp.float32)
    b2 = jnp.array(0.1, dtype=jnp.float32)

    y_base, x_si = ncdm_base(e_s, d_disc, d_kc, q_i, w1, b1, w2, b2)
    x_np = np.asarray(ncdm_interaction(e_s, d_disc, d_kc, q_i))
    hidden = 1.0 / (1.0 + np.exp(-(x_np @ np.asarray(w1) + np.asarray(b1))))
    pre = hidden @ np.asarray(w2) + float(b2)
    y_np = 1.0 / (1.0 + np.exp(-pre))
    np.testing.assert_allclose(np.asarray(x_si), x_np, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(y_base), y_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(ncdm_mlp(x_si, w1, b1, w2, b2)), pre, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(sigmoid(jnp.asarray(pre))), y_np, rtol=1e-5)
