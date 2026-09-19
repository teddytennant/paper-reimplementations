"""minADE@K / minFDE@K on the horizontal (x, y) plane, Eqs. (1)-(2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def ade(pred: Array, gt: Array) -> Array:
    """ADE = (1/T_fut) Σ ||p̂_n - p_n||_2  on (x, y). pred, gt: (T, ≥2)."""
    d = pred[..., :2] - gt[..., :2]
    return jnp.mean(jnp.linalg.norm(d, axis=-1), axis=-1)


def fde(pred: Array, gt: Array) -> Array:
    """FDE = ||p̂_{T_fut} - p_{T_fut}||_2  on (x, y)."""
    d = pred[..., -1, :2] - gt[..., -1, :2]
    return jnp.linalg.norm(d, axis=-1)


def min_ade_at_k(preds: Array, gt: Array) -> Array:
    """min over K samples of ADE. preds: (K, T, ≥2), gt: (T, ≥2)."""
    return jnp.min(jax.vmap(lambda p: ade(p, gt))(preds), axis=0)


def min_fde_at_k(preds: Array, gt: Array) -> Array:
    """min over K samples of FDE. preds: (K, T, ≥2), gt: (T, ≥2)."""
    return jnp.min(jax.vmap(lambda p: fde(p, gt))(preds), axis=0)
