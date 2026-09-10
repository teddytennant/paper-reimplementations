"""Normalized-Laplacian clustering and K selection (Eq. 4–5)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from wrp.constants import EPS, KMEANS_SEED
from wrp.similarity import depth_binned_similarity


def affinity(similarity: Array) -> Array:
    """A = (S + 1) / 2 with unit diagonal."""
    n_layers = similarity.shape[0]
    adj = 0.5 * (similarity + 1.0)
    return adj.at[jnp.diag_indices(n_layers)].set(1.0)


def normalized_laplacian(adj: Array, eps: float = EPS) -> Array:
    """L_sym = I - Delta^{-1/2} A Delta^{-1/2}, Delta = diag(A 1)."""
    n_layers = adj.shape[0]
    degree = adj.sum(axis=1)
    inv_sqrt = 1.0 / jnp.sqrt(jnp.maximum(degree, eps))
    scaled = adj * inv_sqrt[:, None] * inv_sqrt[None, :]
    return jnp.eye(n_layers, dtype=adj.dtype) - scaled


def laplacian_eigengaps(similarity: Array, eps: float = EPS) -> tuple[Array, Array]:
    """Return (d2, d3) = (λ3-λ2, λ4-λ3) of the normalized Laplacian of A(S)."""
    lap = normalized_laplacian(affinity(similarity), eps=eps)
    evals = jnp.linalg.eigh(lap)[0]
    d2 = evals[2] - evals[1]
    d3 = evals[3] - evals[2]
    return d2, d3


def select_num_clusters(similarity: Array, eps: float = EPS) -> tuple[int, float]:
    """Choose K in {2, 4} from eigengaps of S versus depth-binned S0 (Eq. 5)."""
    d2, d3 = laplacian_eigengaps(similarity, eps=eps)
    n2, n3 = laplacian_eigengaps(depth_binned_similarity(similarity), eps=eps)
    e2 = jnp.maximum(d2 - n2, 0.0)
    e3 = jnp.maximum(d3 - n3, 0.0)
    total = e2 + e3
    rho = jnp.where(total > 0, e2 / total, 1.0)
    k = jnp.where((total > 0) & (rho < 0.5), 4, 2)
    return int(k), float(rho)


def _kmeans(points: Array, k: int, seed: int, n_iter: int = 64) -> Array:
    """Lloyd k-means with k-means++ init from a fixed seed."""
    n_points = points.shape[0]
    k = min(int(k), int(n_points))
    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    first = int(jax.random.randint(sub, (), 0, n_points))
    centers = [points[first]]
    for _ in range(1, k):
        stacked = jnp.stack(centers, axis=0)
        dist = jnp.sum((points[:, None, :] - stacked[None, :, :]) ** 2, axis=-1)
        nearest = jnp.min(dist, axis=1)
        key, sub = jax.random.split(key)
        probs = nearest / (jnp.sum(nearest) + EPS)
        idx = int(jax.random.choice(sub, n_points, p=probs))
        centers.append(points[idx])
    centers_arr = jnp.stack(centers, axis=0)
    labels = jnp.zeros((n_points,), dtype=jnp.int32)
    for _ in range(n_iter):
        dist = jnp.sum((points[:, None, :] - centers_arr[None, :, :]) ** 2, axis=-1)
        new_labels = jnp.argmin(dist, axis=1)
        if bool(jnp.array_equal(new_labels, labels)):
            break
        labels = new_labels
        new_centers = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            if int(jnp.sum(mask)) == 0:
                key, sub = jax.random.split(key)
                new_centers.append(points[int(jax.random.randint(sub, (), 0, n_points))])
            else:
                new_centers.append(jnp.mean(points[mask], axis=0))
        centers_arr = jnp.stack(new_centers, axis=0)
    return labels


def spectral_cluster(
    similarity: Array,
    k: int,
    seed: int = KMEANS_SEED,
    eps: float = EPS,
) -> Array:
    """Cluster A into K groups via the normalized-Laplacian embedding and k-means.

    Clusters are then relabeled by increasing depth of their shallowest block.
    """
    adj = affinity(similarity)
    lap = normalized_laplacian(adj, eps=eps)
    evals, evecs = jnp.linalg.eigh(lap)
    del evals
    embedding = evecs[:, :k]
    # Deterministic eigenvector sign: first entry of each column non-negative.
    signs = jnp.where(embedding[0, :] < 0, -1.0, 1.0)
    embedding = embedding * signs[None, :]
    row_norm = jnp.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / jnp.maximum(row_norm, eps)
    labels = _kmeans(embedding, k, seed=seed)
    return _order_by_shallowest(labels, k)


def _order_by_shallowest(labels: Array, k: int) -> Array:
    shallowest = []
    for cluster_id in range(k):
        members = jnp.where(labels == cluster_id)[0]
        if members.size == 0:
            shallowest.append(labels.shape[0] + cluster_id)
        else:
            shallowest.append(int(jnp.min(members)))
    order = jnp.argsort(jnp.array(shallowest))
    remap = jnp.empty((k,), dtype=jnp.int32)
    for new_id, old_id in enumerate(order):
        remap = remap.at[int(old_id)].set(int(new_id))
    return remap[labels]
