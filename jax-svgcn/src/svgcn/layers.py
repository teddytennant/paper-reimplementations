r"""Mask, masked GroupNorm (eq. 8), masked GAP (eq. 10), ST-GCN pieces."""

from __future__ import annotations

import jax
import jax.numpy as jnp

EPS = 1e-5


def make_mask(lengths: jnp.ndarray, t_max: int) -> jnp.ndarray:
    """Binary valid-frame mask M[i,t]=1 iff t < L_i (eq. 7). Shape (N, T)."""
    t = jnp.arange(t_max)
    return (t[None, :] < lengths[:, None]).astype(jnp.float32)


def mask_4d(mask: jnp.ndarray) -> jnp.ndarray:
    """Broadcast ``(N, T)`` to ``(N, 1, T, 1)`` for N,C,T,J tensors."""
    return mask[:, None, :, None]


def downsample_mask(lengths: jnp.ndarray, t_max: int, stride: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Update lengths to ceil(L/s) and rebuild the mask."""
    if stride <= 1:
        return make_mask(lengths, t_max), lengths
    new_t = (t_max + stride - 1) // stride
    new_lengths = (lengths + stride - 1) // stride
    return make_mask(new_lengths, new_t), new_lengths


def masked_group_norm_stats(
    x: jnp.ndarray,
    mask: jnp.ndarray,
    num_groups: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-sample, per-group mean and variance over valid frames (eq. 8).

    ``x`` is ``(N, C, T, J)``, ``mask`` is ``(N, T)``. Returns ``mu, var`` of
    shape ``(N, G)``. Stats are **not** pooled across the batch: the paper's
    sum over n is read as the current sample (text: "within the current
    sample and the current group").
    """
    n, c, t, j = x.shape
    g = num_groups
    if c % g != 0:
        raise ValueError(f"channels {c} not divisible by num_groups {g}")
    cg = c // g
    xg = x.reshape(n, g, cg, t, j)
    m = mask[:, None, None, :, None]
    lengths = jnp.sum(mask, axis=1)
    count = jnp.maximum(lengths * (cg * j), 1.0)[:, None]
    mu = jnp.sum(xg * m, axis=(2, 3, 4)) / count
    centered = xg - mu[:, :, None, None, None]
    var = jnp.sum((centered ** 2) * m, axis=(2, 3, 4)) / count
    return mu, var


def masked_group_norm(
    x: jnp.ndarray,
    mask: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    num_groups: int,
    eps: float = EPS,
) -> jnp.ndarray:
    """Masked GroupNorm with affine ``scale, bias`` of shape ``(C,)``."""
    n, c, t, j = x.shape
    g = num_groups
    cg = c // g
    mu, var = masked_group_norm_stats(x, mask, num_groups)
    xg = x.reshape(n, g, cg, t, j)
    xhat = (xg - mu[:, :, None, None, None]) / jnp.sqrt(var[:, :, None, None, None] + eps)
    xhat = xhat.reshape(n, c, t, j)
    return xhat * scale[:, None, None] + bias[:, None, None]


def masked_gap(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Masked global average pool (eq. 10): Y[n,c] = sum_{t,j} M[n,t] X / max(J L_n, 1)."""
    m = mask_4d(mask)
    j = x.shape[-1]
    lengths = jnp.sum(mask, axis=1)
    denom = jnp.maximum(j * lengths, 1.0)[:, None]
    summed = jnp.sum(x * m, axis=(2, 3))
    return summed / denom


def layer_norm_channels(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, eps: float = EPS) -> jnp.ndarray:
    """LayerNorm over the channel axis of ``(N, C, T, J)``."""
    mu = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.mean((x - mu) ** 2, axis=1, keepdims=True)
    xhat = (x - mu) / jnp.sqrt(var + eps)
    return xhat * scale[:, None, None] + bias[:, None, None]


def linear_nctj(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Channel mix: ``weight`` is ``(C_in, C_out)``, ``x`` is ``(N, C, T, J)``."""
    y = jnp.einsum("nctj,cd->ndtj", x, weight)
    return y + bias[:, None, None]


def spatial_gcn(x: jnp.ndarray, adj: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Graph conv: mix neighbors with normalized A, then 1x1 channel projection."""
    x = jnp.einsum("nctj,jk->nctk", x, adj)
    return linear_nctj(x, weight, bias)


def conv_temporal(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    stride: int = 1,
) -> jnp.ndarray:
    """Temporal conv, kernel (k, 1) on (N, C, T, J) with SAME padding.

    If ``mask`` is given, padded frames are zeroed before the conv so they
    match implicit zero-padding of a truncated sequence, then zeroed again
    after the bias.
    """
    if mask is not None:
        x = x * mask_4d(mask)
    y = jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(stride, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    y = y + bias[:, None, None]
    if mask is not None and stride == 1:
        y = y * mask_4d(mask)
    return y
