"""Rotary position embedding, used only to move between cache and content space.

Section 3.1: source RoPE is inverted before the ridge fit and target RoPE is
applied after mapping, so W and b act on content-space K.
"""

from __future__ import annotations

import jax.numpy as jnp


def rope_angles(positions, head_dim: int, base: float = 10000.0):
    """Rotation angle per half-dimension, shape positions.shape + (head_dim // 2,)."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    inv_freq = base ** (-jnp.arange(half) / half)
    return jnp.asarray(positions)[..., None] * inv_freq


def apply_rope(x, angles):
    """Rotate the halves of x by angles. angles must broadcast against x[..., :d/2]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def invert_rope(x, angles):
    """Undo apply_rope. The rotation is orthogonal, so the inverse is the negated angle."""
    return apply_rope(x, -angles)
