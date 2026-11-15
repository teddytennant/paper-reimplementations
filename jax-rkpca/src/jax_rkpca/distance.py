"""Loadings and half squared subspace distance, paper eqs. (2) and (3)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def loadings(h: Array, mu: Array, Gamma: Array) -> Array:
    """Subspace loadings c = (h - mu)^T Gamma, paper eq. (2).

    Args:
        h: empirical kernel map, shape (M,).
        mu: running mean in feature space, shape (M,).
        Gamma: orthonormal basis, shape (M, d).

    Returns:
        Loadings of shape (d,).
    """
    h = jnp.asarray(h)
    mu = jnp.asarray(mu)
    Gamma = jnp.asarray(Gamma)
    return (h - mu) @ Gamma


def half_squared_distance(h: Array, mu: Array, Gamma: Array) -> Array:
    """Half squared distance from the centered feature to the subspace.

    Paper eq. (3):

        z(h, mu, Gamma) = (1/2) || (I - Gamma Gamma^T)(h - mu) ||^2

    With orthonormal columns this equals (1/2)(||x||^2 - ||Gamma^T x||^2)
    where x = h - mu. The residual form is used so the value stays correct
    even if Gamma is only approximately orthonormal.
    """
    h = jnp.asarray(h)
    mu = jnp.asarray(mu)
    Gamma = jnp.asarray(Gamma)
    x = h - mu
    # residual = (I - P) x = x - Gamma (Gamma^T x), never form I or P
    proj = Gamma @ (Gamma.T @ x)
    residual = x - proj
    return 0.5 * jnp.dot(residual, residual)
