"""Additive linearity, multiplicative bilinearity, relational attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rivat_fuse.interaction import (
    init_interaction_params,
    interaction,
    relational_weights,
    t_add,
    t_mult,
    t_rel,
)


D = 8
S_DIM = 3
K = 2
R_K = 4


def _params(seed=0):
    return init_interaction_params(jax.random.PRNGKey(seed), D, S_DIM, K=K, r_k=R_K)


def test_additive_linearity_in_each_modality():
    params = _params(0)
    key = jax.random.PRNGKey(1)
    z_I = jax.random.normal(key, (D,))
    z_M = jax.random.normal(jax.random.fold_in(key, 1), (D,))
    a = 2.3
    # Unconditioned additive is linear in each argument.
    t_a = t_add(params, a * z_I, z_M, s=None)
    t_1 = t_add(params, z_I, z_M, s=None)
    t_scale_I = t_add(params, a * z_I, jnp.zeros_like(z_M), s=None)
    t_I_only = t_add(params, z_I, jnp.zeros_like(z_M), s=None)
    assert jnp.allclose(t_scale_I, a * t_I_only, atol=1e-5)
    t_scale_M = t_add(params, jnp.zeros_like(z_I), a * z_M, s=None)
    t_M_only = t_add(params, jnp.zeros_like(z_I), z_M, s=None)
    assert jnp.allclose(t_scale_M, a * t_M_only, atol=1e-5)
    # Superposition: T_add(z_I, 0) + T_add(0, z_M) = T_add(z_I, z_M).
    assert jnp.allclose(
        t_add(params, z_I, jnp.zeros_like(z_M), s=None)
        + t_add(params, jnp.zeros_like(z_I), z_M, s=None),
        t_add(params, z_I, z_M, s=None),
        atol=1e-5,
    )
    # Homogeneity in z_I with z_M fixed: T(a z_I, z_M) = a A_I z_I + A_M z_M.
    expected = a * t_add(params, z_I, jnp.zeros_like(z_M), s=None) + t_add(
        params, jnp.zeros_like(z_I), z_M, s=None
    )
    assert jnp.allclose(t_a, expected, atol=1e-5)
    assert not jnp.allclose(t_a, t_1, atol=1e-4)


def test_multiplicative_hadamard_bilinear():
    params = _params(2)
    key = jax.random.PRNGKey(3)
    z_I = jax.random.normal(key, (D,))
    z_M = jax.random.normal(jax.random.fold_in(key, 1), (D,))
    zeros = jnp.zeros((D,))
    assert jnp.allclose(t_mult(params, zeros, z_M, s=None), zeros, atol=1e-6)
    assert jnp.allclose(t_mult(params, z_I, zeros, s=None), zeros, atol=1e-6)
    a, c = 1.7, -0.4
    left = t_mult(params, a * z_I, c * z_M, s=None)
    right = (a * c) * t_mult(params, z_I, z_M, s=None)
    assert jnp.allclose(left, right, atol=1e-5)


def test_relational_pi_sums_to_one():
    params = _params(4)
    key = jax.random.PRNGKey(5)
    L = 5
    z_M = jax.random.normal(key, (D,))
    tokens = jax.random.normal(jax.random.fold_in(key, 1), (L, D))
    pi = relational_weights(params, z_M, tokens)
    assert pi.shape == (L,)
    assert jnp.allclose(jnp.sum(pi), 1.0, atol=1e-5)
    assert jnp.all(pi >= 0)


def test_relational_peak_on_huge_match():
    params = _params(6)
    key = jax.random.PRNGKey(7)
    L = 4
    z_M = jax.random.normal(key, (D,))
    query = params["W_q"] @ z_M
    tokens = 0.01 * jax.random.normal(jax.random.fold_in(key, 1), (L, D))
    # Token 2 is a huge match to W_q z_M.
    tokens = tokens.at[2].set(50.0 * query / (jnp.linalg.norm(query) + 1e-8))
    pi = relational_weights(params, z_M, tokens)
    assert float(pi[2]) > 0.99
    assert float(jnp.sum(pi) - 1.0) < 1e-5


def test_t_rel_in_span_of_tokens_after_W_R():
    params = _params(8)
    key = jax.random.PRNGKey(9)
    L = 4
    z_M = jax.random.normal(key, (D,))
    tokens = jax.random.normal(jax.random.fold_in(key, 1), (L, D))
    t, pi = t_rel(params, z_M, tokens)
    pooled = pi @ tokens
    assert jnp.allclose(t, params["W_R"] @ pooled, atol=1e-5)
    # t lives in the column span of W_R (equivalently W_R applied to span of tokens).
    # Least-squares residual of solving W_R x = t should be ~0.
    x, *_ = jnp.linalg.lstsq(params["W_R"], t, rcond=None)
    assert jnp.allclose(params["W_R"] @ x, t, atol=1e-5)


def test_t_rel_zero_without_tokens():
    params = _params(10)
    z_M = jnp.ones((D,))
    t, pi = t_rel(params, z_M, None)
    assert pi is None
    assert jnp.allclose(t, jnp.zeros((D,)))


def test_combine_is_sum_of_parts():
    params = _params(11)
    key = jax.random.PRNGKey(12)
    z_I = jax.random.normal(key, (D,))
    z_M = jax.random.normal(jax.random.fold_in(key, 1), (D,))
    tokens = jax.random.normal(jax.random.fold_in(key, 2), (3, D))
    s = jax.random.normal(jax.random.fold_in(key, 3), (S_DIM,))
    t = interaction(params, z_I, z_M, tokens=tokens, s=s)
    t_r, _ = t_rel(params, z_M, tokens)
    expected = t_add(params, z_I, z_M, s=s) + t_mult(params, z_I, z_M, s=s) + t_r
    assert jnp.allclose(t, expected, atol=1e-5)
