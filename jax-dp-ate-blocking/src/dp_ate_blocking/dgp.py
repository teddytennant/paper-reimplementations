"""Synthetic DGPs from arXiv:2609.09536 §4.3."""

from __future__ import annotations

import jax
import jax.numpy as jnp


_MEAN_12 = jnp.array([2.0, 4.0])
_COV = jnp.array([[1.0, 0.2], [0.2, 1.0]])
_CLIP = jnp.array([4.5, 6.5, 4.5, 6.5])


def _logit_well_specified(x):
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return 0.4 - x1 + 0.5 * x2 + 0.025 * x2**2 - 0.25 * x3 - 0.1 * x4


def _logit_low_overlap(x):
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return 1.5 - 2.0 * x1 + x2 + 0.05 * x2**2 - 0.5 * x3 - 0.2 * x4


def _mu(x, w, tau):
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return tau * w + 0.1 * x1 - 0.05 * x2 + 0.05 * x3 - 0.05 * x4


def _sample_covariates(key, n):
    k1, k2 = jax.random.split(key)
    x12 = jax.random.multivariate_normal(k1, _MEAN_12, _COV, (n,))
    x34 = jax.random.multivariate_normal(k2, _MEAN_12, _COV, (n,))
    return jnp.concatenate([x12, x34], axis=1)


def _clip_and_scale(x):
    x = jnp.clip(x, -_CLIP, _CLIP)
    scale = 1.0 / jnp.linalg.norm(_CLIP)
    return x * scale


def well_specified(key, n, tau=2.0, setting="well"):
    """Well-specified (or low-overlap) DGP from §4.3.

    Returns (x, w, y, e_true) with x clipped and scaled to the unit ball.
    Outcomes are translated by τ/2 then clipped to |Y| <= (τ+1)/2 so the
    ATE remains τ when clipping is rare.
    """
    key_x, key_w, key_y = jax.random.split(key, 3)
    x_raw = _sample_covariates(key_x, n)
    if setting == "low_overlap":
        logits = _logit_low_overlap(x_raw)
    else:
        logits = _logit_well_specified(x_raw)
    e_true = jax.nn.sigmoid(logits)
    u = jax.random.uniform(key_w, (n,))
    w = (u < e_true).astype(jnp.float64)
    mu = _mu(x_raw, w, tau)
    y = mu + jax.random.normal(key_y, (n,)) * jnp.sqrt(0.02)
    # Paper: "translate then clip" so |Y| <= (τ+1)/2.
    y = y - tau / 2.0
    c_y = (tau + 1.0) / 2.0
    y = jnp.clip(y, -c_y, c_y)
    x = _clip_and_scale(x_raw)
    return x, w, y, e_true
