"""Transient 3D grid used by LPS and LHA.

Neighbors of p_i are every point whose integer cell is within Chebyshev
distance `radius` of p_i's cell. With radius=1 this is the home cell plus
the 26 adjacent cells. Not kNN.
"""

import jax
import jax.numpy as jnp


def cell_indices(
    points: jax.Array,
    cell_size: float | jax.Array,
    origin: jax.Array | None = None,
) -> jax.Array:
    """Integer cell coordinates, shape (N, 3)."""
    cell_size = jnp.asarray(cell_size, dtype=points.dtype)
    if origin is None:
        origin = jnp.zeros((3,), dtype=points.dtype)
    return jnp.floor((points - origin) / cell_size).astype(jnp.int32)


def neighbor_mask(
    points: jax.Array,
    cell_size: float | jax.Array,
    origin: jax.Array | None = None,
    radius: int = 1,
) -> jax.Array:
    """(N, N) bool mask. True iff the two points are transient-grid neighbors.

    Includes self (same cell, Chebyshev distance 0).
    """
    cells = cell_indices(points, cell_size, origin)
    diff = jnp.abs(cells[:, None, :] - cells[None, :, :])
    return jnp.all(diff <= radius, axis=-1)
