"""Left truncated SVD projector (GaLore-style row projection, Algorithm 1)."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def truncated_svd(V: Array, rank: int, key=None) -> Array:
    """Return left projector P ∈ R^{m×r} from a truncated SVD of V.

    Uses ``jax.numpy.linalg.svd(..., full_matrices=False)``. ``key`` is accepted
    for API compatibility and is unused (dense SVD, not randomized).
    """
    del key
    V = jnp.asarray(V)
    if V.ndim != 2:
        raise ValueError(f"V must be 2-D, got shape {V.shape}")
    u, _s, _vh = jnp.linalg.svd(V, full_matrices=False)
    r = int(rank)
    r = max(0, min(r, int(u.shape[1])))
    return u[:, :r]


def project(V: Array, P: Array) -> Array:
    """Ṽ = P^T V ∈ R^{r×n}."""
    return jnp.matmul(jnp.asarray(P).T, jnp.asarray(V))


def reconstruct_direction(dV_tilde: Array, P: Array) -> Array:
    """ΔV = P ΔṼ ∈ R^{m×n}."""
    return jnp.matmul(jnp.asarray(P), jnp.asarray(dV_tilde))
