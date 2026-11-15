"""Empirical kernel map, paper eq. (1)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

_KERNELS = ("rbf", "linear")


def rbf_kernel(x: Array, y: Array, sigma: float = 1.0) -> Array:
    """RBF kernel k(x, y) = exp(-||x-y||^2 / (2 sigma^2))."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    diff = x - y
    return jnp.exp(-jnp.dot(diff, diff) / (2.0 * sigma * sigma))


def linear_kernel(x: Array, y: Array) -> Array:
    """Linear kernel k(x, y) = x^T y."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return jnp.dot(x, y)


def empirical_kernel_map(
    x: Array,
    U: Array,
    kernel: str = "rbf",
    sigma: float = 1.0,
) -> Array:
    """Finite-basis empirical kernel map, paper eq. (1).

    h = [k(u_1, x), ..., k(u_M, x)]^T

    Args:
        x: incoming sample, shape (D,).
        U: basis points, shape (M, D).
        kernel: ``"rbf"`` or ``"linear"``.
        sigma: RBF bandwidth (ignored for the linear kernel).

    Returns:
        Feature vector h of shape (M,).
    """
    x = jnp.asarray(x)
    U = jnp.asarray(U)
    if U.ndim != 2:
        raise ValueError("U must have shape (M, D)")
    if x.ndim != 1 or x.shape[0] != U.shape[1]:
        raise ValueError("x must have shape (D,) matching U")
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}, got {kernel!r}")

    if kernel == "linear":
        return U @ x

    diff = U - x[None, :]
    sq = jnp.sum(diff * diff, axis=1)
    return jnp.exp(-sq / (2.0 * sigma * sigma))
