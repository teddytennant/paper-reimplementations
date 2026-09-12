"""Dense and axis-restricted attention on a C x T token grid.

Tokens are indexed by i = (c_i, t_i) on Omega = C x T. Dense attention
mixes every pair. The temporal path attends only within a channel; the
spatial path attends only within a time step. Optional k-NN restriction
on the spatial path uses 3D electrode coordinates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

NEG_INF = -1e9


def init_linear(key, in_dim, out_dim):
    return jax.random.normal(key, (in_dim, out_dim)) * (in_dim**-0.5)


def init_attention_params(key, d):
    """Q, K, V, O projections in R^{d x d} for one path."""
    kq, kk, kv, ko = jax.random.split(key, 4)
    return {
        "Wq": init_linear(kq, d, d),
        "Wk": init_linear(kk, d, d),
        "Wv": init_linear(kv, d, d),
        "Wo": init_linear(ko, d, d),
    }


def _split_heads(x, num_heads):
    b, c, t, d = x.shape
    dh = d // num_heads
    return x.reshape(b, c, t, num_heads, dh)


def _merge_heads(x):
    b, c, t, h, dh = x.shape
    return x.reshape(b, c, t, h * dh)


def _project_qkv(params, x, num_heads):
    q = _split_heads(x @ params["Wq"], num_heads)
    k = _split_heads(x @ params["Wk"], num_heads)
    v = _split_heads(x @ params["Wv"], num_heads)
    return q, k, v


def _scale(dh):
    return dh**0.5


def temporal_token_mask(c, t):
    """Boolean mask (C, T, C, T): True iff query and key share a channel."""
    c_idx = jnp.arange(c)
    same_c = c_idx[:, None] == c_idx[None, :]
    return jnp.broadcast_to(same_c[:, None, :, None], (c, t, c, t))


def spatial_token_mask(c, t):
    """Boolean mask (C, T, C, T): True iff query and key share a time step."""
    t_idx = jnp.arange(t)
    same_t = t_idx[:, None] == t_idx[None, :]
    return jnp.broadcast_to(same_t[None, :, None, :], (c, t, c, t))


def knn_spatial_mask(positions, k):
    """Electrode k-NN mask (C, C).

    A query electrode attends to itself and its ``k`` nearest neighbours
    under Euclidean distance on 3D coordinates ``positions`` of shape (C, 3).
    ``k=None`` keeps every electrode.
    """
    c = positions.shape[0]
    if k is None:
        return jnp.ones((c, c), dtype=bool)
    delta = positions[:, None, :] - positions[None, :, :]
    dist = jnp.linalg.norm(delta, axis=-1)
    order = jnp.argsort(dist, axis=-1)
    keep = order[:, : k + 1]
    mask = jnp.zeros((c, c), dtype=bool)
    rows = jnp.arange(c)[:, None]
    mask = mask.at[rows, keep].set(True)
    mask = mask.at[jnp.arange(c), jnp.arange(c)].set(True)
    return mask


def dense_attention(params, x, num_heads, return_weights=False):
    """Full-grid attention: softmax over all C*T keys for each query.

    x: (B, C, T, D) -> (B, C, T, D). Weights, if requested, are
    (B, H, C, T, C, T) and sum to 1 over the last two axes.
    """
    b, c, t, d = x.shape
    dh = d // num_heads
    q, k, v = _project_qkv(params, x, num_heads)
    q = jnp.transpose(q, (0, 3, 1, 2, 4)).reshape(b, num_heads, c * t, dh)
    k = jnp.transpose(k, (0, 3, 1, 2, 4)).reshape(b, num_heads, c * t, dh)
    v = jnp.transpose(v, (0, 3, 1, 2, 4)).reshape(b, num_heads, c * t, dh)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / _scale(dh)
    weights_flat = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights_flat, v)
    out = out.reshape(b, num_heads, c, t, dh)
    out = jnp.transpose(out, (0, 2, 3, 1, 4))
    out = _merge_heads(out) @ params["Wo"]
    if return_weights:
        weights = weights_flat.reshape(b, num_heads, c, t, c, t)
        return out, weights
    return out


def temporal_attention(params, x, num_heads, return_weights=False):
    """Attend along time within each channel. Neighbourhood |T(i)| = T.

    Weights, if requested, have shape (B, H, C, T, C, T) with mass only
    on keys that share the query channel.
    """
    b, c, t, d = x.shape
    dh = d // num_heads
    q, k, v = _project_qkv(params, x, num_heads)
    q = jnp.transpose(q, (0, 1, 3, 2, 4))
    k = jnp.transpose(k, (0, 1, 3, 2, 4))
    v = jnp.transpose(v, (0, 1, 3, 2, 4))
    scores = jnp.einsum("bchqd,bchkd->bchqk", q, k) / _scale(dh)
    w_tt = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bchqk,bchkd->bchqd", w_tt, v)
    out = jnp.transpose(out, (0, 1, 3, 2, 4))
    out = _merge_heads(out) @ params["Wo"]
    if return_weights:
        eye_c = jnp.eye(c)
        weights = jnp.einsum("bchtk,cd->bhctdk", w_tt, eye_c)
        return out, weights
    return out


def spatial_attention(params, x, num_heads, knn_mask=None, return_weights=False):
    """Attend across channels at a fixed time. Neighbourhood |S(i)| = C.

    ``knn_mask`` is an optional (C, C) boolean from :func:`knn_spatial_mask`.
    Weights, if requested, have shape (B, H, C, T, C, T) with mass only
    on keys that share the query time (and the k-NN set, if given).
    """
    b, c, t, d = x.shape
    dh = d // num_heads
    q, k, v = _project_qkv(params, x, num_heads)
    q = jnp.transpose(q, (0, 2, 3, 1, 4))
    k = jnp.transpose(k, (0, 2, 3, 1, 4))
    v = jnp.transpose(v, (0, 2, 3, 1, 4))
    scores = jnp.einsum("bthqd,bthkd->bthqk", q, k) / _scale(dh)
    if knn_mask is not None:
        scores = jnp.where(knn_mask[None, None, None, :, :], scores, NEG_INF)
    w_cc = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bthqk,bthkd->bthqd", w_cc, v)
    out = jnp.transpose(out, (0, 3, 1, 2, 4))
    out = _merge_heads(out) @ params["Wo"]
    if return_weights:
        eye_t = jnp.eye(t)
        weights = jnp.einsum("bthck,ts->bhctks", w_cc, eye_t)
        return out, weights
    return out
