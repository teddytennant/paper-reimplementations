"""Adaptive-bandwidth RBF kernel on Euclidean token embeddings."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def pairwise_squared_euclidean(embeddings: Array) -> Array:
    """Pairwise squared Euclidean distances, shape (n, n)."""
    embeddings = jnp.asarray(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings[:, None]
    sq = jnp.sum(embeddings * embeddings, axis=-1)
    gram = embeddings @ embeddings.T
    sqdist = sq[:, None] + sq[None, :] - 2.0 * gram
    return jnp.maximum(sqdist, 0.0)


def pairwise_euclidean(embeddings: Array) -> Array:
    """Pairwise Euclidean distances, shape (n, n)."""
    return jnp.sqrt(pairwise_squared_euclidean(embeddings))


def adaptive_bandwidth(embeddings: Array) -> Array:
    """Median pairwise Euclidean distance over i < j.

    ε_t = median_{i<j} ||e_i - e_j||_2.  Returns 0 when n < 2 or when
    every pairwise distance is 0 (identical embeddings).
    """
    embeddings = jnp.asarray(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings[:, None]
    n = int(embeddings.shape[0])
    if n < 2:
        return jnp.zeros((), dtype=embeddings.dtype)
    dist = pairwise_euclidean(embeddings)
    ii, jj = jnp.triu_indices(n, k=1)
    return jnp.median(dist[ii, jj])


def rbf_kernel(embeddings: Array, epsilon: Array | float) -> Array:
    """Gaussian RBF on Euclidean distances.

    K_ij = exp( - ||e_i - e_j||^2 / (2 ε_t^2) ).
    When ε_t = 0 the kernel is the all-ones matrix.
    """
    embeddings = jnp.asarray(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings[:, None]
    sq = pairwise_squared_euclidean(embeddings)
    eps = jnp.asarray(epsilon, dtype=sq.dtype)
    eps2 = eps * eps
    denom = jnp.where(eps2 == 0, jnp.ones_like(eps2), eps2)
    k = jnp.exp(-sq / (2.0 * denom))
    return jnp.where(eps2 == 0, jnp.ones_like(k), k)
