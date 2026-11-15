"""Numerically stable primitives used by ARACD.

LayerNorm (eq. 15) is last-axis population normalization with no affine
scale/bias:

    LN(x) = (x - mean) / sqrt(var + eps)

where mean and var are taken over the last axis (ddof=0). After LN, that
axis has mean 0 and variance 1 when var >> eps.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import nn

PROB_EPS = 1e-6
LN_EPS = 1e-5


def sigmoid(x):
    return nn.sigmoid(x)


def clip_prob(p, eps: float = PROB_EPS):
    """Clip probabilities away from {0, 1} before logit (eq. 7)."""
    return jnp.clip(p, eps, 1.0 - eps)


def logit(p, eps: float = PROB_EPS):
    p = clip_prob(p, eps)
    return jnp.log(p) - jnp.log1p(-p)


def layer_norm(x, eps: float = LN_EPS):
    """Last-axis population LayerNorm, no affine parameters."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)
