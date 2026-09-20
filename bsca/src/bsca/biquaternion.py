"""Biquaternion algebra: Hamilton product, Hermitian norm, inner product.

A biquaternion is q = w + x i + y j + z k with w, x, y, z in C.
Embeddings are complex arrays of shape (..., 4, k) for components (w, x, y, z)
and k independent slots. The Hamilton product is slot-wise along the 4-axis.
"""

from __future__ import annotations

import jax.numpy as jnp

_COMPONENT_AXIS = -2


def _as_bq(q):
    q = jnp.asarray(q)
    if q.ndim == 0:
        raise ValueError("biquaternion must have shape (..., 4, k)")
    if q.shape[-1] == 4 and (q.ndim == 1 or q.shape[-2] != 4):
        q = q[..., None]
    if q.ndim < 2 or q.shape[_COMPONENT_AXIS] != 4:
        raise ValueError(f"expected (..., 4, k), got {q.shape}")
    return q


def hamilton(q1, q2):
    """Slot-wise Hamilton product (paper eq. 5).

    q1 q2 =
      (w1 w2 - x1 x2 - y1 y2 - z1 z2)
    + (w1 x2 + x1 w2 + y1 z2 - z1 y2) i
    + (w1 y2 - x1 z2 + y1 w2 + z1 x2) j
    + (w1 z2 + x1 y2 - y1 x2 + z1 w2) k
    """
    q1 = _as_bq(q1)
    q2 = _as_bq(q2)
    w1, x1, y1, z1 = jnp.moveaxis(q1, _COMPONENT_AXIS, 0)
    w2, x2, y2, z2 = jnp.moveaxis(q2, _COMPONENT_AXIS, 0)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.stack([w, x, y, z], axis=_COMPONENT_AXIS)


def bq_norm(q):
    """Hermitian norm (paper eq. 6): sqrt(|w|^2 + |x|^2 + |y|^2 + |z|^2) over slots."""
    q = _as_bq(q)
    return jnp.sqrt(jnp.sum(jnp.abs(q) ** 2, axis=(-2, -1)))


def bq_inner(q1, q2):
    """Real part of the Hermitian inner product over the 4k complex components."""
    q1 = _as_bq(q1)
    q2 = _as_bq(q2)
    return jnp.real(jnp.sum(jnp.conj(q1) * q2, axis=(-2, -1)))


def one_like(q):
    """Multiplicative identity 1 + 0i + 0j + 0k, same shape as q."""
    q = _as_bq(q)
    out = jnp.zeros_like(q)
    return out.at[..., 0, :].set(jnp.ones(q.shape[:-2] + (q.shape[-1],), dtype=q.dtype))
