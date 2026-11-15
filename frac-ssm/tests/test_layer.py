"""Read/write softmax, timescale prior, AR cache, layer shapes."""

import jax
import jax.numpy as jnp

from frac_ssm.bank import geometric_timescales
from frac_ssm.layer import (
    apply_layer,
    decode_autoregressive,
    init_frac_layer,
)
from frac_ssm.transition import read_write_weights


def test_softmax_sums_to_one_and_short_tau_prior():
    tau = geometric_timescales(0.5, 2.0, 4)
    g = jnp.zeros((4,))
    alpha_small = jnp.array(0.2)
    alpha_large = jnp.array(0.8)
    c_small, b_small = read_write_weights(alpha_small, tau, g, g)
    c_large, b_large = read_write_weights(alpha_large, tau, g, g)
    assert jnp.allclose(jnp.sum(c_small), 1.0)
    assert jnp.allclose(jnp.sum(b_small), 1.0)
    assert jnp.allclose(jnp.sum(c_large), 1.0)
    assert jnp.allclose(jnp.sum(b_large), 1.0)
    # Larger α puts more mass on the shortest timescale (smallest τ).
    assert c_large[0] > c_small[0]
    assert c_large[-1] < c_small[-1]
    assert b_large[0] > b_small[0]


def test_layer_shapes():
    key = jax.random.PRNGKey(0)
    B, T, D, H, M = 2, 7, 16, 2, 4
    params = init_frac_layer(key, d=D, n_heads=H, n_modes=M)
    x = jax.random.normal(jax.random.PRNGKey(1), (B, T, D))
    y, S = apply_layer(params, x, parallel=True)
    dh = D // H
    assert y.shape == (B, T, D)
    assert S.shape == (B, H, M, dh)


def test_ar_cached_matches_parallel_prefill():
    key = jax.random.PRNGKey(2)
    B, T, D = 2, 9, 16
    params = init_frac_layer(key, d=D, n_heads=2, n_modes=4)
    x = jax.random.normal(jax.random.PRNGKey(3), (B, T, D))
    y_par, _ = apply_layer(params, x, parallel=True)
    y_ar = decode_autoregressive(params, x)
    assert y_par.shape == y_ar.shape
    assert jnp.allclose(y_par, y_ar, atol=1e-5, rtol=1e-5)


def test_layer_parallel_matches_sequential_mixer():
    key = jax.random.PRNGKey(4)
    B, T, D = 2, 8, 16
    params = init_frac_layer(key, d=D)
    x = jax.random.normal(jax.random.PRNGKey(5), (B, T, D))
    y_par, S_par = apply_layer(params, x, parallel=True)
    y_seq, S_seq = apply_layer(params, x, parallel=False)
    assert jnp.allclose(y_par, y_seq, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(S_par, S_seq, atol=1e-5, rtol=1e-5)
