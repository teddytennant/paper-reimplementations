"""Euclidean projections onto common convex sets."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def project_unconstrained(x) -> Array:
    """Identity projector (X = R^n or Y = R^m)."""
    return jnp.asarray(x)


def project_box(x, lo, hi) -> Array:
    """Componentwise clip onto [lo, hi] (scalars or arrays)."""
    return jnp.clip(jnp.asarray(x), lo, hi)


def project_ball(x, radius: float = 1.0, center=None) -> Array:
    """Euclidean ball of given radius about ``center`` (default origin)."""
    x = jnp.asarray(x)
    if center is None:
        center = jnp.zeros_like(x)
    else:
        center = jnp.asarray(center, dtype=x.dtype)
    d = x - center
    nrm = jnp.linalg.norm(d)
    scale = jnp.where(nrm > radius, radius / jnp.maximum(nrm, 1e-32), 1.0)
    return center + d * scale


def project_simplex(v, s: float = 1.0) -> Array:
    """Euclidean projection onto {x ≥ 0, 1^T x = s} (Duchi et al. 2008)."""
    v = jnp.asarray(v)
    n = v.shape[0]
    u = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(u) - s
    ind = jnp.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = jnp.maximum(jnp.count_nonzero(cond), 1)
    theta = cssv[rho - 1] / rho
    return jnp.maximum(v - theta, 0.0)
