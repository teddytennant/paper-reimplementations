"""Isotropic KDN sequential step (eq 40, Proposition C.1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def isotropic_predict_scale(
    b: jax.Array | float,
    alpha: jax.Array,
    omega: jax.Array | float,
) -> tuple[jax.Array, jax.Array]:
    """a = mean(α²), b̂ = a b + ω."""
    a = jnp.mean(alpha * alpha)
    b_hat = a * b + omega
    return a, b_hat


def isotropic_beta(b_hat: jax.Array | float, k: jax.Array, r: jax.Array | float) -> jax.Array:
    """β = b̂ / (r + b̂ ||k||²)."""
    k2 = jnp.dot(k, k)
    return b_hat / (r + b_hat * k2)


def isotropic_step(
    S: jax.Array,
    b: jax.Array | float,
    alpha: jax.Array,
    omega: jax.Array | float,
    k: jax.Array,
    v: jax.Array,
    r: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One Isotropic KDN step.

    Memory: S = (I - β k k^T) D S + β k v^T with D = diag(α).
    Scale: b ← b̂ / (1 + b̂ ||k||² / (d_k r)) from the isotropic reverse-KL projection.
    """
    _, b_hat = isotropic_predict_scale(b, alpha, omega)
    beta = isotropic_beta(b_hat, k, r)
    S_hat = alpha[:, None] * S
    innov = v - jnp.dot(jnp.swapaxes(S_hat, -1, -2), k)
    S_new = S_hat + beta * jnp.outer(k, innov)
    d_k = alpha.shape[-1]
    k2 = jnp.dot(k, k)
    b_new = b_hat / (1.0 + b_hat * k2 / (d_k * r))
    return S_new, b_new, beta


def isotropic_filter(
    S0: jax.Array,
    b0: jax.Array | float,
    alpha: jax.Array,
    omega: jax.Array,
    k: jax.Array,
    v: jax.Array,
    r: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sequential isotropic filter.

    Args:
        S0: (d_k, d_v), b0: scalar.
        alpha: (T, d_k), omega: (T,), k: (T, d_k), v: (T, d_v), r: (T,).
    """

    def body(carry, inputs):
        S, b = carry
        a_t, w_t, k_t, v_t, r_t = inputs
        S, b, beta = isotropic_step(S, b, a_t, w_t, k_t, v_t, r_t)
        return (S, b), (S, b, beta)

    _, (S, b, beta) = lax.scan(body, (S0, jnp.asarray(b0, dtype=S0.dtype)), (alpha, omega, k, v, r))
    return S, b, beta
