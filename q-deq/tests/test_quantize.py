"""Lock eq 7 / eq 8 W8A8 grid, scale, and STE of round."""

import jax
import jax.numpy as jnp

from q_deq.quantize import (
    QMAX,
    QMIN,
    activation_scale,
    quantize_activation,
    quantize_weight,
    quantize_with_scale,
    ste_round,
    weight_scale,
)


def test_eq7_weight_scale_and_grid():
    key = jax.random.PRNGKey(8)
    W = jax.random.normal(key, (6, 4)) * 3.0
    Wq, s = quantize_weight(W)
    s_ref = jnp.max(jnp.abs(W), axis=0) / 127.0
    assert jnp.allclose(s, s_ref, atol=1e-6)
    assert jnp.allclose(s, weight_scale(W))
    q = Wq / s
    assert jnp.allclose(q, jnp.round(q), atol=1e-5)
    assert jnp.all(q >= QMIN - 1e-5)
    assert jnp.all(q <= QMAX + 1e-5)
    # Reconstruct: dequantized values sit on the 8-bit grid.
    assert jnp.allclose(Wq, s * jnp.clip(jnp.round(W / s), QMIN, QMAX), atol=1e-5)


def test_eq8_activation_scale_and_grid():
    key = jax.random.PRNGKey(9)
    A = jax.random.normal(key, (5, 7)) * 2.5
    Aq, s = quantize_activation(A)
    s_ref = jnp.max(jnp.abs(A)) / 127.0
    assert jnp.allclose(s, s_ref, atol=1e-6)
    assert jnp.allclose(s, activation_scale(A))
    q = Aq / s
    assert jnp.allclose(q, jnp.round(q), atol=1e-5)
    assert jnp.all(q >= QMIN - 1e-5) and jnp.all(q <= QMAX + 1e-5)


def test_ste_round_gradient_is_identity_inside_clip():
    x = jnp.array([-3.2, -0.4, 0.0, 1.7, 10.3], dtype=jnp.float32)
    g_round = jax.grad(lambda t: ste_round(t).sum())(x)
    assert jnp.allclose(g_round, jnp.ones_like(x), atol=1e-6)

    # Fixed scale, values well inside (-127, 127) after /s.
    s = jnp.array(1.0, dtype=jnp.float32)
    g = jax.grad(lambda t: quantize_with_scale(t, s).sum())(x)
    assert jnp.allclose(g, jnp.ones_like(x), atol=1e-5)

    # Outside the clip range the clip STE is zero.
    x_out = jnp.array([200.0, -200.0], dtype=jnp.float32)
    g_out = jax.grad(lambda t: quantize_with_scale(t, s).sum())(x_out)
    assert jnp.allclose(g_out, jnp.zeros_like(x_out), atol=1e-5)
