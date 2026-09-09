"""Fixed-gain delta-rule special case (eq 18-20): κ = β k."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def isotropic_gain_scale(b_hat: jax.Array | float, k: jax.Array, r: jax.Array | float) -> jax.Array:
    """β = b̂ / (r + b̂ ||k||²) from the isotropic surrogate P̂ ≈ b̂ I."""
    k2 = jnp.dot(k, k)
    return b_hat / (r + b_hat * k2)


def kappa_from_beta(beta: jax.Array | float, k: jax.Array) -> jax.Array:
    """κ = β k."""
    return beta * k


def delta_rule_step(
    S: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta: jax.Array | float,
    D: jax.Array | None = None,
    alpha: jax.Array | None = None,
) -> jax.Array:
    """S = (I - β k k^T) D S + β k v^T.

    Pass ``D`` as a dense matrix, or ``alpha`` for D = diag(α). Default D = I.
    """
    if D is not None:
        S_hat = D @ S
    elif alpha is not None:
        S_hat = alpha[:, None] * S
    else:
        S_hat = S
    innov = v - jnp.dot(jnp.swapaxes(S_hat, -1, -2), k)
    return S_hat + beta * jnp.outer(k, innov)
