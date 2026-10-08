"""Token-wise pruning bound from the theory section.

This is the theorem, not the DIPrune inference score. At inference, gradients
are unavailable, so DIPrune replaces ``w_i`` with
``alpha * S_align + beta * S_evol``.
"""

import jax.numpy as jnp
import numpy as np


def _as_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _token_matrices(v, g):
    v = jnp.asarray(v)
    g = jnp.asarray(g)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if g.ndim == 1:
        g = g.reshape(-1, 1)
    if v.ndim != 2 or g.ndim != 2:
        raise ValueError("v and g must have shape (N, D)")
    if v.shape != g.shape:
        raise ValueError("v and g must share shape (N, D)")
    if v.shape[0] < 1:
        raise ValueError("v must contain at least one token")
    return v, g


def _dropped(mask, n, dtype):
    mask = jnp.asarray(mask)
    if mask.shape != (n,):
        raise ValueError(f"mask must have shape ({n},)")
    return 1.0 - mask.astype(dtype)


def inner_bound(v, g, mask, l_f):
    """Descent bound after the token decomposition, before Young.

    ``B_inner(m) = sum_i (1 - m_i) * (|<g_i, v_i>| + (L_F / 2) * ||v_i||^2)``.
    """
    v, g = _token_matrices(v, g)
    dropped = _dropped(mask, v.shape[0], v.dtype)
    inner = jnp.sum(g * v, axis=-1)
    v_sq = jnp.sum(v * v, axis=-1)
    per_token = jnp.abs(inner) + (l_f / 2.0) * v_sq
    return jnp.sum(dropped * per_token)


def token_weights(v, g, mu, l_f):
    """Relaxed per-token weight from the Young step.

    ``w_i = (1 / (2 * mu)) * ||g_i||^2 + ((mu + L_F) / 2) * ||v_i||^2``.
    ``mu`` must be positive. These weights are the theorem's selection
    criterion. They are not what DIPrune computes at inference.
    """
    try:
        mu_value = float(mu)
    except TypeError as exc:
        raise ValueError("mu must be a positive real number") from exc
    if mu_value <= 0:
        raise ValueError("mu must be positive")
    v, g = _token_matrices(v, g)
    g_sq = jnp.sum(g * g, axis=-1)
    v_sq = jnp.sum(v * v, axis=-1)
    return (1.0 / (2.0 * mu_value)) * g_sq + ((mu_value + l_f) / 2.0) * v_sq


def relaxed_bound(v, g, mask, mu, l_f):
    """``B_relaxed(m) = sum_i (1 - m_i) * w_i``."""
    v, g = _token_matrices(v, g)
    weights = token_weights(v, g, mu, l_f)
    dropped = _dropped(mask, v.shape[0], weights.dtype)
    return jnp.sum(dropped * weights)


def topk_mask(weights, k):
    """Binary mask that keeps the K largest weights. 1 means keep.

    Among masks with exactly K ones, this choice minimizes the relaxed bound.
    Ties keep the smaller index. ``k`` must lie in ``1..N``.
    """
    weights = jnp.asarray(weights)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D vector")
    n = int(weights.shape[0])
    if n < 1:
        raise ValueError("weights must be non-empty")
    k = _as_int(k, "k")
    if k < 1 or k > n:
        raise ValueError(f"k must be in 1..{n}, got {k}")
    index = jnp.arange(n)
    order = jnp.lexsort((index, -weights))
    mask = jnp.zeros(n, dtype=jnp.int32)
    return mask.at[order[:k]].set(1)
