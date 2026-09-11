"""Gaussian (RBF) Mercer kernel with K(x, x) = 1."""

from __future__ import annotations

import jax.numpy as jnp


def gaussian_kernel(x, y, lengthscale: float = 1.0):
    """K(x, y) = exp(−‖x−y‖² / (2 ℓ²)). Broadcasts over leading batch axes."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    diff = x - y
    sq = jnp.sum(diff * diff, axis=-1)
    return jnp.exp(-sq / (2.0 * lengthscale**2))


def kernel_matrix(X, Y=None, lengthscale: float = 1.0):
    """Gram matrix with rows of X against rows of Y (or X)."""
    X = jnp.asarray(X)
    Y = X if Y is None else jnp.asarray(Y)
    # (n, 1, d) - (1, m, d) -> (n, m, d)
    diff = X[:, None, :] - Y[None, :, :]
    sq = jnp.sum(diff * diff, axis=-1)
    return jnp.exp(-sq / (2.0 * lengthscale**2))
