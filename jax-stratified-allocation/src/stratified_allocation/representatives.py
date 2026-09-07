"""Proposition 7 of arXiv:2609.04420, deterministic cluster representatives.

The fourth construction rule is not a probability sample. Class c is partitioned into
K_c clusters in feature space, one representative r(C) is retained per cluster and given
the weight w_{r(C)} = |C|. Because the clustering never looks at l and l changes at every
boosting round, there is no unbiasedness statement to be had; Equation (10) says exactly
what the error is.
"""

from typing import Tuple

import jax
import jax.numpy as jnp


def kmeans_labels(
    key: jax.Array,
    features: jax.Array,
    n_clusters: int,
    iters: int = 50,
    init: jax.Array = None,
) -> jax.Array:
    """Plain Lloyd's algorithm from a random-row initialisation.

    The clustering is formed in feature space with no reference to the losses, which is
    the assumption Proposition 7 rests on. Pass ``init`` (row indices for the starting
    centres) when you need the same partition every time; Lloyd's answer depends on
    where it starts, and Proposition 7 says nothing about which partition you get.
    """
    n = features.shape[0]
    if init is None:
        init = jax.random.permutation(key, n)[:n_clusters]
    centres = features[jnp.asarray(init)]

    def step(centres, _):
        d = jnp.sum((features[:, None, :] - centres[None, :, :]) ** 2, axis=-1)
        assign = jnp.argmin(d, axis=1)
        onehot = (assign[:, None] == jnp.arange(n_clusters)[None, :]).astype(features.dtype)
        counts = jnp.sum(onehot, axis=0)
        sums = jnp.einsum("nk,nd->kd", onehot, features)
        new = jnp.where(counts[:, None] > 0, sums / jnp.maximum(counts, 1)[:, None], centres)
        return new, assign

    centres, assigns = jax.lax.scan(step, centres, None, length=iters)
    return assigns[-1]


def cluster_representatives(features: jax.Array, assignment: jax.Array, n_clusters: int) -> jax.Array:
    """The medoid of each cluster: the row closest to its cluster's centroid.

    This is the deterministic r(C) of Proposition 7. Empty clusters return index -1.
    """
    onehot = (assignment[:, None] == jnp.arange(n_clusters)[None, :]).astype(features.dtype)
    counts = jnp.sum(onehot, axis=0)
    sums = jnp.einsum("nk,nd->kd", onehot, features)
    centres = sums / jnp.maximum(counts, 1)[:, None]
    d = jnp.sum((features[:, None, :] - centres[None, :, :]) ** 2, axis=-1)
    d = jnp.where(onehot > 0, d, jnp.inf)
    reps = jnp.argmin(d, axis=0)
    return jnp.where(counts > 0, reps, -1)


def representative_estimate(
    losses: jax.Array, assignment: jax.Array, representatives: jax.Array, n_clusters: int
) -> jax.Array:
    """Lhat = sum_C |C| l_{r(C)}, the retained representative weighted by cluster size."""
    onehot = (assignment[:, None] == jnp.arange(n_clusters)[None, :]).astype(losses.dtype)
    sizes = jnp.sum(onehot, axis=0)
    keep = representatives >= 0
    picked = jnp.where(keep, losses[jnp.maximum(representatives, 0)], 0.0)
    return jnp.sum(jnp.where(keep, sizes, 0.0) * picked)


def representative_bias(
    losses: jax.Array, assignment: jax.Array, representatives: jax.Array, n_clusters: int
) -> jax.Array:
    """Equation (10): Lhat - L = sum_c sum_C |C| (l_{r(C)} - lbar_C).

    It vanishes for all Phi only if l is constant within every cluster. There is no
    expectation anywhere in this statement; the design is deterministic given the
    clustering, so the error is the bias.
    """
    onehot = (assignment[:, None] == jnp.arange(n_clusters)[None, :]).astype(losses.dtype)
    sizes = jnp.sum(onehot, axis=0)
    sums = jnp.sum(onehot * losses[:, None], axis=0)
    means = sums / jnp.maximum(sizes, 1)
    keep = representatives >= 0
    picked = jnp.where(keep, losses[jnp.maximum(representatives, 0)], 0.0)
    return jnp.sum(jnp.where(keep, sizes * (picked - means), 0.0))


def representative_design(
    key: jax.Array,
    features: jax.Array,
    losses: jax.Array,
    n_clusters: int,
    iters: int = 50,
) -> Tuple[jax.Array, jax.Array]:
    """Cluster, retain medoids, return (estimate, bias). Convenience wrapper."""
    assignment = kmeans_labels(key, features, n_clusters, iters)
    reps = cluster_representatives(features, assignment, n_clusters)
    return (
        representative_estimate(losses, assignment, reps, n_clusters),
        representative_bias(losses, assignment, reps, n_clusters),
    )
