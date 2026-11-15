"""Online attention with EFQ current-block weights (Algorithm 2)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from efq_softmax.quantize import (
    DEFAULT_BLOCK_SIZE,
    H_MMLU,
    TAU_MMLU,
    apply_operand,
    efq_lut,
    efq_quantize,
)


def softmax_attention(
    q: Array,
    k: Array,
    v: Array,
    mask: Array | None = None,
) -> Array:
    """Exact one-shot softmax attention (FlashAttention reference)."""
    scale = jnp.sqrt(jnp.asarray(q.shape[-1], dtype=q.dtype))
    scores = (q @ jnp.swapaxes(k, -1, -2)) / scale
    if mask is not None:
        scores = jnp.where(mask, scores, -jnp.inf)
    shifted = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(shifted)
    denom = jnp.sum(weights, axis=-1, keepdims=True)
    weights = jnp.where(denom > 0, weights / denom, jnp.zeros_like(weights))
    return weights @ v


def _current_weights(
    x: Array,
    *,
    exact: bool,
    lut: bool,
    tau: float,
    h: float,
    block_size: int,
) -> Array:
    if exact:
        return jnp.exp(jnp.where(jnp.isfinite(x), x, -jnp.inf))
    if lut:
        return efq_lut(x, block_size=block_size).ptilde
    return efq_quantize(x, tau=tau, h=h, block_size=block_size).ptilde


def online_attention(
    q: Array,
    k: Array,
    v: Array,
    *,
    tile_k: int | None = None,
    tau: float = TAU_MMLU,
    h: float = H_MMLU,
    block_size: int = DEFAULT_BLOCK_SIZE,
    lut: bool = False,
    exact: bool = False,
    mask: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Tiled online attention (Algorithm 2).

    Historical rescaling uses a row-level ``exp`` (eq 5). Current-block
    ``exp`` is replaced by EFQ unless ``exact=True``.

    Returns ``(output, numerator A, denominator l)``.
    """
    nq = q.shape[-2]
    d = q.shape[-1]
    nk = k.shape[-2]
    dv = v.shape[-1]
    tile = nk if tile_k is None else tile_k
    scale = jnp.sqrt(jnp.asarray(d, dtype=q.dtype))

    batch_shape = q.shape[:-2]
    m = jnp.full((*batch_shape, nq, 1), -jnp.inf, dtype=q.dtype)
    a = jnp.zeros((*batch_shape, nq, dv), dtype=q.dtype)
    ell = jnp.zeros((*batch_shape, nq, 1), dtype=q.dtype)

    n_tiles = (nk + tile - 1) // tile
    for j in range(n_tiles):
        start = j * tile
        stop = min(start + tile, nk)
        k_j = k[..., start:stop, :]
        v_j = v[..., start:stop, :]
        scores = (q @ jnp.swapaxes(k_j, -1, -2)) / scale
        if mask is not None:
            scores = jnp.where(mask[..., start:stop], scores, -jnp.inf)
        rowmax = jnp.max(scores, axis=-1, keepdims=True)
        m_new = jnp.maximum(m, rowmax)
        x = scores - m_new
        ptilde = _current_weights(
            x, exact=exact, lut=lut, tau=tau, h=h, block_size=block_size
        )
        alpha = jnp.exp(m - m_new)
        alpha = jnp.where(jnp.isfinite(alpha), alpha, jnp.zeros_like(alpha))
        delta_a, delta_l = apply_operand(ptilde, v_j)
        a = alpha * a + delta_a
        ell = alpha * ell + delta_l
        m = m_new

    out = jnp.where(ell > 0, a / ell, jnp.zeros_like(a))
    return out, a, ell
