"""Pairwise intensities of the dynamic latent-space Poisson process (eq 2.1)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def squared_distance(z_i: Array, z_j: Array) -> Array:
    """||z_i - z_j||^2, broadcasting over a leading time axis if present."""
    diff = z_i - z_j
    return jnp.sum(diff * diff, axis=-1)


def log_intensity(
    beta_i: Array,
    beta_j: Array,
    z_i: Array,
    z_j: Array,
    model: str = "full",
) -> Array:
    """log λ_ij(t) from eq 2.1, or a nested restriction.

    full:        log λ_ij = β_i + β_j − ||z_i − z_j||²
    homogeneous: same formula; constancy in t is a property of static z
    beta_only:   log λ_ij = β_i + β_j  (distance term dropped)
    """
    base = beta_i + beta_j
    if model == "beta_only":
        return base
    return base - squared_distance(z_i, z_j)


def intensity(
    beta_i: Array,
    beta_j: Array,
    z_i: Array,
    z_j: Array,
    model: str = "full",
) -> Array:
    """λ_ij(t) = exp(β_i + β_j − d_ij(t)²) (eq 2.1)."""
    return jnp.exp(log_intensity(beta_i, beta_j, z_i, z_j, model=model))


def pairwise_squared_distances(z: Array) -> Array:
    """All-pairs squared distances for a configuration z of shape (n, 2) or (n, M, 2)."""
    if z.ndim == 2:
        diff = z[:, None, :] - z[None, :, :]
        return jnp.sum(diff * diff, axis=-1)
    if z.ndim == 3:
        diff = z[:, None, :, :] - z[None, :, :, :]
        return jnp.sum(diff * diff, axis=-1)
    raise ValueError(f"expected z of rank 2 or 3, got {z.ndim}")


def pairwise_log_intensity(beta: Array, z: Array, model: str = "full") -> Array:
    """log λ_ij for all ordered pairs, shape (n, n) or (n, n, M)."""
    n = beta.shape[0]
    bi = beta[:, None]
    bj = beta[None, :]
    if z.ndim == 3:
        bi = bi[:, :, None]
        bj = bj[:, :, None]
    base = bi + bj
    if model == "beta_only":
        if z.ndim == 3:
            return jnp.broadcast_to(base, (n, n, z.shape[1]))
        return base
    return base - pairwise_squared_distances(z)


def apply_rigid_transform(z: Array, translation: Array, rotation: Array) -> Array:
    """Map z ↦ (z − translation) @ rotation. Distances (hence λ) are invariant."""
    return (z - translation) @ rotation
