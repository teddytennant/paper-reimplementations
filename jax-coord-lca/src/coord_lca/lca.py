"""Localized Cross-Attention (LCA).

Map each query onto the encoder token grid (eq. 7), take the 2x2 neighborhood
of tokens whose centers surround the mapped location, and cross-attend:

    f_q' = CA(f_q, F_local)

``F_local`` has size ``k = 4``. An optional global token (mean of all encoder
tokens, or a dedicated CLS) may be concatenated as an extra key/value.

Bilinear 2x2
------------
Token ``(i, j)`` is centered at encoder-grid coordinate ``(j + 0.5, i + 0.5)``.
The four surrounding tokens of ``(x_down, y_down)`` have integer indices

    j0 = floor(x_down - 0.5),  i0 = floor(y_down - 0.5)

with bilinear weights on the shifted coordinates ``(x_down - 0.5, y_down - 0.5)``.
At the exact center of a token the weights concentrate on that token.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from coord_lca.patch import init_linear, linear


def bilinear_neighborhood(
    xy_enc: jnp.ndarray,
    *,
    encoder_height: int,
    encoder_width: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """2x2 neighborhood indices and bilinear weights.

    Parameters
    ----------
    xy_enc:
        ``(..., 2)`` encoder-grid coordinates ``(x_down, y_down)`` from eq. 7.

    Returns
    -------
    indices:
        ``(..., 4, 2)`` integer ``(row, col)`` of the four tokens in order
        ``(i0, j0), (i0, j1), (i1, j0), (i1, j1)``.
    weights:
        ``(..., 4)`` bilinear weights summing to 1 (interior points).
    xy_shifted:
        ``(..., 2)`` ``(x_down - 0.5, y_down - 0.5)`` used by the formula.
    """
    xy_enc = jnp.asarray(xy_enc, dtype=jnp.float32)
    x = xy_enc[..., 0]
    y = xy_enc[..., 1]
    x_c = x - 0.5
    y_c = y - 0.5
    j0 = jnp.floor(x_c)
    i0 = jnp.floor(y_c)
    wx = x_c - j0
    wy = y_c - i0
    j0_i = j0.astype(jnp.int32)
    i0_i = i0.astype(jnp.int32)
    j1_i = j0_i + 1
    i1_i = i0_i + 1

    j0_c = jnp.clip(j0_i, 0, encoder_width - 1)
    j1_c = jnp.clip(j1_i, 0, encoder_width - 1)
    i0_c = jnp.clip(i0_i, 0, encoder_height - 1)
    i1_c = jnp.clip(i1_i, 0, encoder_height - 1)

    idx00 = jnp.stack([i0_c, j0_c], axis=-1)
    idx01 = jnp.stack([i0_c, j1_c], axis=-1)
    idx10 = jnp.stack([i1_c, j0_c], axis=-1)
    idx11 = jnp.stack([i1_c, j1_c], axis=-1)
    indices = jnp.stack([idx00, idx01, idx10, idx11], axis=-2)

    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy
    weights = jnp.stack([w00, w01, w10, w11], axis=-1)
    xy_shifted = jnp.stack([x_c, y_c], axis=-1)
    return indices, weights, xy_shifted


def gather_neighborhood(
    tokens: jnp.ndarray,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    """Gather encoder tokens at integer ``(row, col)`` indices.

    Parameters
    ----------
    tokens:
        ``(H_e, W_e, D)``.
    indices:
        ``(N, 4, 2)`` or ``(N, 2)``.

    Returns
    -------
    ``(N, 4, D)`` or ``(N, D)``.
    """
    height, width, dim = tokens.shape
    flat = tokens.reshape(height * width, dim)
    rows = indices[..., 0]
    cols = indices[..., 1]
    lin = rows * width + cols
    lin = jnp.clip(lin, 0, height * width - 1)
    gathered = flat[lin]
    return gathered


def bilinear_sample_tokens(
    tokens: jnp.ndarray,
    xy_enc: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Gather the 2x2 neighborhood and the bilinear weights at ``xy_enc``."""
    height, width = tokens.shape[:2]
    indices, weights, _ = bilinear_neighborhood(
        xy_enc, encoder_height=height, encoder_width=width
    )
    neighborhood = gather_neighborhood(tokens, indices)
    return neighborhood, weights, indices


def init_lca(
    key: jax.Array,
    *,
    d_model: int,
    n_heads: int = 4,
) -> dict:
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    kq, kk, kv, ko = jax.random.split(key, 4)
    return {
        "W_q": init_linear(kq, d_model, d_model),
        "W_k": init_linear(kk, d_model, d_model),
        "W_v": init_linear(kv, d_model, d_model),
        "W_o": init_linear(ko, d_model, d_model),
    }


def cross_attend(
    params: dict,
    query: jnp.ndarray,
    kv: jnp.ndarray,
    *,
    n_heads: int = 4,
) -> jnp.ndarray:
    """Multi-head cross-attention: ``query`` ``(N, D)``, ``kv`` ``(N, K, D)``.

    Returns ``(N, D)``. Independent of any dense ``H*W`` map.
    """
    n_queries, dim = query.shape
    if dim % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    k_len = kv.shape[1]
    head_dim = dim // n_heads
    q = linear(params["W_q"], query).reshape(n_queries, 1, n_heads, head_dim)
    k = linear(params["W_k"], kv).reshape(n_queries, k_len, n_heads, head_dim)
    v = linear(params["W_v"], kv).reshape(n_queries, k_len, n_heads, head_dim)
    scale = head_dim**-0.5
    logits = jnp.einsum("nqhd,nkhd->nhqk", q, k) * scale
    attn = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("nhqk,nkhd->nqhd", attn, v).reshape(n_queries, dim)
    return linear(params["W_o"], out)


def localized_cross_attention(
    params: dict,
    query: jnp.ndarray,
    tokens: jnp.ndarray,
    xy_enc: jnp.ndarray,
    *,
    include_global: bool = True,
    global_token: jnp.ndarray | None = None,
    n_heads: int = 4,
) -> jnp.ndarray:
    """LCA: 2x2 local keys/values, optional global token, output ``(N, D)``.

    Parameters
    ----------
    query:
        Coordinate-aware queries ``f_q``, shape ``(N, D)``.
    tokens:
        Encoder grid ``F_g``, shape ``(H_e, W_e, D)``.
    xy_enc:
        Mapped encoder-grid coordinates ``(N, 2)``.
    include_global:
        If true, append a global key/value (mean of ``tokens``, or
        ``global_token`` if given).
    """
    neighborhood, _, _ = bilinear_sample_tokens(tokens, xy_enc)
    kv = neighborhood
    if include_global:
        if global_token is None:
            global_token = tokens.mean(axis=(0, 1))
        g = jnp.broadcast_to(global_token, (query.shape[0], global_token.shape[-1]))
        kv = jnp.concatenate([kv, g[:, None, :]], axis=1)
    return cross_attend(params, query, kv, n_heads=n_heads)
