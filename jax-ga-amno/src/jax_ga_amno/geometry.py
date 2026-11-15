"""Reference grid, neighborhoods, local Jacobians, and edge descriptors.

Local mesh Jacobian J^a_i is a least-squares fit of relative adaptive
coordinates onto relative reference coordinates over the neighborhood:

    Δx^a ≈ J Δx^r
    J = (ΔX^a)^T (ΔX^r)  ((ΔX^r)^T (ΔX^r))^{+}

On a uniform grid with zero displacement this recovers I_2 in the interior.
The paper says Jacobians are "estimated"; this estimator is ours.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def reference_grid(
    height: int,
    width: int,
    omega_min: float = 0.0,
    omega_max: float = 1.0,
) -> Array:
    """Node coordinates in Ω, shape (H*W, 2), ordered row-major as (x, y)."""
    ys = jnp.linspace(omega_min, omega_max, height)
    xs = jnp.linspace(omega_min, omega_max, width)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([xx, yy], axis=-1).reshape(height * width, 2)


def four_connected_neighbors(height: int, width: int) -> Array:
    """(N, 4) indices: left, right, down, up. Borders clamp (one-sided)."""
    n = height * width
    y = jnp.arange(n) // width
    x = jnp.arange(n) % width
    left = y * width + jnp.clip(x - 1, 0, width - 1)
    right = y * width + jnp.clip(x + 1, 0, width - 1)
    down = jnp.clip(y - 1, 0, height - 1) * width + x
    up = jnp.clip(y + 1, 0, height - 1) * width + x
    return jnp.stack([left, right, down, up], axis=-1).astype(jnp.int32)


def knn_neighbors(x_r: Array, k: int) -> Array:
    """k-NN in reference coordinates, excluding self. x_r: (N, D) -> (N, k)."""
    diff = x_r[:, None, :] - x_r[None, :, :]
    d2 = jnp.sum(diff * diff, axis=-1)
    d2 = d2 + jnp.eye(x_r.shape[0], dtype=d2.dtype) * 1e6
    return jax.lax.top_k(-d2, k)[1].astype(jnp.int32)


def neighbor_mask(neighbors: Array) -> Array:
    """True where the neighbor is not a clamped self-loop."""
    n = neighbors.shape[0]
    return neighbors != jnp.arange(n, dtype=neighbors.dtype)[:, None]


def local_jacobians(x_a: Array, x_r: Array, neighbors: Array) -> Array:
    """Least-squares J^a_i, shape (N, D, D)."""
    x_a_j = x_a[neighbors]
    x_r_j = x_r[neighbors]
    b = x_a_j - x_a[:, None, :]
    a = x_r_j - x_r[:, None, :]
    ata = jnp.einsum("nkd,nke->nde", a, a)
    bta = jnp.einsum("nkd,nke->nde", b, a)
    return bta @ jnp.linalg.pinv(ata)


def edge_descriptors(
    x_a: Array,
    jacobians: Array,
    state: Array,
    neighbors: Array,
) -> Array:
    """e_ij = Concat(x^a_j - x^a_i, vec(J^a_j), vec(J^a_i), p_j, p_i)  (Eq. 7)."""
    n, k = neighbors.shape
    dim = x_a.shape[-1]
    rel = x_a[neighbors] - x_a[:, None, :]
    j_j = jacobians[neighbors].reshape(n, k, dim * dim)
    j_i = jnp.broadcast_to(
        jacobians.reshape(n, 1, dim * dim), (n, k, dim * dim)
    )
    p_j = state[neighbors]
    p_i = jnp.broadcast_to(state[:, None, :], p_j.shape)
    return jnp.concatenate([rel, j_j, j_i, p_j, p_i], axis=-1)
