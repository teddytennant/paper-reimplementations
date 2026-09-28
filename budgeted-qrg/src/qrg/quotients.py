"""Toy quotient charts q and Jacobians J_q = Dq.

Pairwise distance is rotation- and translation-invariant. The contact chart is a
logistic of that distance (a pairwise pocket-style feature).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pairwise_distance(x, i: int = 0, j: int = 1, eps: float = 1e-12):
    """q = ||x_i - x_j|| with J shape (1, n*3)."""
    x = jnp.asarray(x)
    n = x.shape[0]
    diff = x[i] - x[j]
    dist = jnp.linalg.norm(diff)
    unit = jnp.where(dist > eps, diff / dist, jnp.zeros_like(diff))
    jac = jnp.zeros((n, 3), dtype=x.dtype)
    jac = jac.at[i].set(unit)
    jac = jac.at[j].set(-unit)
    return dist, jac.reshape(1, n * 3)


def pairwise_contact(
    x,
    i: int = 0,
    j: int = 1,
    cutoff: float = 4.0,
    steepness: float = 1.0,
    eps: float = 1e-12,
):
    """Soft contact q = σ(k (r0 - d)) with J shape (1, n*3)."""
    dist, jac_d = pairwise_distance(x, i=i, j=j, eps=eps)
    q = jax.nn.sigmoid(steepness * (cutoff - dist))
    dqd = -steepness * q * (1.0 - q)
    return q, dqd * jac_d
