"""Convex combination of temporal and spatial paths."""

import jax
import jax.numpy as jnp

from aaa.gate import (
    fixed_lambda,
    fixed_mix,
    gated_mix,
    init_fixed_gate,
    init_token_gate,
    token_gate_weights,
)


def test_gated_weights_nonnegative_and_sum_to_one():
    key = jax.random.PRNGKey(0)
    d = 16
    params = init_token_gate(key, d)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 5, d))
    alpha, beta = token_gate_weights(params, x, tau=1.0)
    assert alpha.shape == (2, 3, 5, 1)
    assert beta.shape == (2, 3, 5, 1)
    assert bool(jnp.all(alpha >= 0.0))
    assert bool(jnp.all(beta >= 0.0))
    assert jnp.allclose(alpha + beta, 1.0, atol=1e-6)


def test_gated_zero_init_is_uniform():
    params = init_token_gate(jax.random.PRNGKey(2), 8)
    x = jax.random.normal(jax.random.PRNGKey(3), (1, 2, 2, 8))
    alpha, beta = token_gate_weights(params, x, tau=1.0)
    assert jnp.allclose(alpha, 0.5, atol=1e-6)
    assert jnp.allclose(beta, 0.5, atol=1e-6)


def test_fixed_lambda_is_sigmoid_and_shared():
    params = init_fixed_gate(ell=0.0)
    lam = fixed_lambda(params)
    assert jnp.allclose(lam, 0.5)
    params_hi = init_fixed_gate(ell=2.0)
    lam_hi = float(fixed_lambda(params_hi))
    assert 0.0 < lam_hi < 1.0
    assert lam_hi > 0.8
    temporal = jnp.ones((2, 4, 6, 3))
    spatial = jnp.zeros((2, 4, 6, 3))
    mixed = fixed_mix(params_hi, temporal, spatial)
    # same lambda for every token
    assert jnp.allclose(mixed, lam_hi)
    assert mixed.shape == temporal.shape


def test_fixed_weights_nonnegative_sum_to_one_per_layer():
    params = init_fixed_gate(ell=-0.7)
    lam = float(fixed_lambda(params))
    assert lam >= 0.0
    assert (1.0 - lam) >= 0.0
    assert abs(lam + (1.0 - lam) - 1.0) < 1e-6


def test_gated_mix_is_convex_combination():
    key = jax.random.PRNGKey(4)
    d = 8
    params = init_token_gate(key, d)
    # perturb the output layer so the mix is not 1/2
    params = dict(params)
    params["W2"] = params["W2"] + 0.4
    x = jax.random.normal(jax.random.PRNGKey(5), (1, 2, 3, d))
    temporal = jax.random.normal(jax.random.PRNGKey(6), x.shape)
    spatial = jax.random.normal(jax.random.PRNGKey(7), x.shape)
    mixed = gated_mix(params, x, temporal, spatial, tau=0.5)
    alpha, beta = token_gate_weights(params, x, tau=0.5)
    expected = alpha * temporal + beta * spatial
    assert jnp.allclose(mixed, expected, atol=1e-6)
