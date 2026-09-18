"""Differentiable batch kNN Local Outlier Factor on latent z (Eqs 2–4, 11)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pairwise_euclidean(z: jax.Array) -> jax.Array:
    diff = z[:, None, :] - z[None, :, :]
    return jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-12)


def lof_scores(z: jax.Array, k: int = 5) -> jax.Array:
    """Per-point LOF_k. Inliers in a homogeneous cluster score near 1; outliers >> 1.

    kNN is hard (argsort). Gradients flow through the taken distances (Eqs 2–4).
    """
    n = z.shape[0]
    k = int(k)
    k = min(k, max(n - 1, 1))
    dist = pairwise_euclidean(z)
    dist = dist + jnp.eye(n, dtype=z.dtype) * 1.0e6
    knn_idx = jnp.argsort(dist, axis=1)[:, :k]
    knn_dist = jnp.take_along_axis(dist, knn_idx, axis=1)
    k_dist = knn_dist[:, -1]
    k_dist_j = k_dist[knn_idx]
    reach = jnp.maximum(knn_dist, k_dist_j)
    lrd = 1.0 / (jnp.mean(reach, axis=1) + 1e-12)
    lrd_j = lrd[knn_idx]
    return jnp.mean(lrd_j / (lrd[:, None] + 1e-12), axis=1)


def mean_lof(z: jax.Array, k: int = 5) -> jax.Array:
    """Eq 11: L_LOF(z) = (1/n) sum_i LOF_k(z_i). Goal LOF ≈ 1."""
    return jnp.mean(lof_scores(z, k=k))
