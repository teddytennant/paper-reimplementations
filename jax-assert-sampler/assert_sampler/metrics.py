"""Distributional error for the toy problem.

The paper reports FID on image datasets. FID is meaningless in 2D, so the same
role is played here by the energy distance, a proper metric between distributions
that is estimated from samples with no fitted Gaussian in between.
"""

from __future__ import annotations

import jax.numpy as jnp


def _mean_pairwise(a, b, exclude_diagonal=False):
    d = jnp.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    if exclude_diagonal:
        n = a.shape[0]
        return (jnp.sum(d) - jnp.trace(d)) / (n * (n - 1))
    return jnp.mean(d)


def energy_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """E = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||.

    Zero if and only if the two distributions coincide, and nonnegative otherwise.
    """
    return (2.0 * _mean_pairwise(x, y)
            - _mean_pairwise(x, x, True) - _mean_pairwise(y, y, True))


def moment_error(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Max absolute difference of first and second moments, a cheap sanity check."""
    dm = jnp.abs(jnp.mean(x, 0) - jnp.mean(y, 0))
    dc = jnp.abs(jnp.cov(x.T) - jnp.cov(y.T))
    return jnp.maximum(jnp.max(dm), jnp.max(dc))


def cosine(a, b, axis=-1, eps=1e-12):
    na = jnp.linalg.norm(a, axis=axis)
    nb = jnp.linalg.norm(b, axis=axis)
    return jnp.sum(a * b, axis=axis) / jnp.maximum(na * nb, eps)
