"""Weight-space similarity (Eq. 1–3) of arXiv:2609.09883."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from wrp.constants import EPS, PROJ_KEYS, SCALE_KEYS

WeightBlock = Mapping[str, Array]
WeightStack = Sequence[WeightBlock]


def center_output_channels(weight: Array) -> Array:
    """Apply H_d = I_d - (1/d) 11^T to the output (row) axis of W in R^{d x m}."""
    return weight - jnp.mean(weight, axis=0, keepdims=True)


def gram(weight: Array) -> Array:
    """Centered Gram G = (H_d W) (H_d W)^T in R^{d x d} (Eq. 1)."""
    centered = center_output_channels(weight)
    return centered @ centered.T


def linear_cka(g_i: Array, g_j: Array, eps: float = EPS) -> Array:
    """Linear CKA between two Gram matrices (Eq. 2)."""
    inner = jnp.sum(g_i * g_j)
    denom = jnp.linalg.norm(g_i) * jnp.linalg.norm(g_j) + eps
    return inner / denom


def _cka_matrix(grams: Array, eps: float = EPS) -> Array:
    """All-pairs linear CKA for a stack of Grams with shape (L, d, d)."""
    flat = grams.reshape((grams.shape[0], -1))
    inner = flat @ flat.T
    norms = jnp.linalg.norm(flat, axis=1)
    return inner / (norms[:, None] * norms[None, :] + eps)


def projection_scale_vectors(weights: WeightStack) -> Array:
    """f_ell = (||W_ell^a||_F)_a for a in the seven projections, shape (L, 7)."""
    rows = []
    for block in weights:
        rows.append(jnp.stack([jnp.linalg.norm(block[name]) for name in SCALE_KEYS]))
    return jnp.stack(rows, axis=0)


def pairwise_similarity(weights: WeightStack, eps: float = EPS) -> Array:
    """Build S from projection CKA and centered scale cosine (Eq. 2–3)."""
    n_layers = len(weights)
    proj_ckas = []
    for key in PROJ_KEYS:
        grams = jnp.stack([gram(block[key]) for block in weights], axis=0)
        proj_ckas.append(_cka_matrix(grams, eps=eps))
    s_proj = 0.5 * (proj_ckas[0] + proj_ckas[1])

    scale = projection_scale_vectors(weights)
    scale_centered = scale - jnp.mean(scale, axis=0, keepdims=True)
    scale_inner = scale_centered @ scale_centered.T
    scale_norms = jnp.linalg.norm(scale_centered, axis=1)
    s_scale = scale_inner / (scale_norms[:, None] * scale_norms[None, :] + eps)

    similarity = 0.5 * (s_proj + s_scale)
    similarity = 0.5 * (similarity + similarity.T)
    similarity = similarity.at[jnp.diag_indices(n_layers)].set(1.0)
    return similarity


def depth_binned_similarity(similarity: Array) -> Array:
    """S0: replace off-diagonal S_ij by the mean of S at lag |i-j|."""
    n_layers = similarity.shape[0]
    s0 = jnp.eye(n_layers, dtype=similarity.dtype)
    idx = jnp.arange(n_layers)
    lag = jnp.abs(idx[:, None] - idx[None, :])
    for distance in range(1, n_layers):
        mask = lag == distance
        mean_val = jnp.sum(similarity * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        s0 = jnp.where(mask, mean_val, s0)
    s0 = s0.at[jnp.diag_indices(n_layers)].set(1.0)
    return s0
