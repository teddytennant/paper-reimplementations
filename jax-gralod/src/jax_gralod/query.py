"""Continuous LOD query (Eq. 4--5, 12--15)."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from jax_gralod.ops import Array


def clamp_lod(lam: Array, n_levels: int) -> Array:
    """lambda in [0, L-1]."""
    return jnp.clip(lam, 0.0, float(n_levels - 1))


def triangular_weights(lam: Array, n_levels: int) -> Array:
    """omega_l(lambda) = max(0, 1 - |lambda - l|) (Eq. 13). Output (..., L)."""
    levels = jnp.arange(n_levels, dtype=lam.dtype)
    return jnp.maximum(0.0, 1.0 - jnp.abs(lam[..., None] - levels))


def neighboring_levels(lam: Array, n_levels: int) -> tuple[Array, Array, Array]:
    """l^- = floor(lambda), l^+ = ceil(lambda), alpha = lambda - l^- (Eq. 5)."""
    lam = clamp_lod(lam, n_levels)
    l_minus = jnp.floor(lam)
    l_plus = jnp.minimum(l_minus + 1.0, float(n_levels - 1))
    # At integers ceil(k)=k; floor(k)+1 would be k+1, so snap when alpha==0.
    alpha = lam - l_minus
    l_plus = jnp.where(alpha <= 0.0, l_minus, l_plus)
    return l_minus, l_plus, alpha


def lod_query_lerp(lam: Array, z_levels: Sequence[Array]) -> Array:
    """Two-level lerp (Eq. 5 / 12). Floor has zero JAX grad, so dF/d lambda = Z^{l+1}-Z^l."""
    z = jnp.stack(tuple(z_levels), axis=-2)  # (..., L, C)
    n_levels = z.shape[-2]
    lam = clamp_lod(lam, n_levels)
    squeezed = lam
    if squeezed.shape[-1] == 1:
        squeezed = squeezed[..., 0]
    l_minus, l_plus, alpha = neighboring_levels(squeezed, n_levels)
    alpha = alpha[..., None]
    idx0 = l_minus.astype(jnp.int32)[..., None, None]
    idx1 = l_plus.astype(jnp.int32)[..., None, None]
    z0 = jnp.take_along_axis(z, idx0, axis=-2)[..., 0, :]
    z1 = jnp.take_along_axis(z, idx1, axis=-2)[..., 0, :]
    return (1.0 - alpha) * z0 + alpha * z1


def lod_query(lam: Array, z_levels: Sequence[Array]) -> Array:
    """F_LOD = sum_l omega_l(lambda) Z^l (Eq. 13), equivalent to the two-level lerp.

    Implemented as a weighted sum so that for lambda in (l, l+1),
    d F / d lambda = Z^{l+1} - Z^l (Eq. 15) without stop-gradient on floor.
    """
    z = jnp.stack(tuple(z_levels), axis=-2)  # (..., L, C)
    n_levels = z.shape[-2]
    squeezed = lam
    if squeezed.shape[-1] == 1:
        squeezed = squeezed[..., 0]
    squeezed = clamp_lod(squeezed, n_levels)
    w = triangular_weights(squeezed, n_levels)
    return jnp.sum(w[..., None] * z, axis=-2)
