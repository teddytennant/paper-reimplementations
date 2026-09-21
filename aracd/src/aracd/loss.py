"""Supervised losses: BCE on responses, optional MSE on affect labels."""

from __future__ import annotations

import jax.numpy as jnp

from aracd.ops import clip_prob


def binary_cross_entropy(y_hat, y, eps: float = 1e-7):
    """Mean BCE of ŷ against binary y."""
    y_hat = clip_prob(y_hat, eps)
    return -jnp.mean(y * jnp.log(y_hat) + (1.0 - y) * jnp.log1p(-y_hat))


def affect_mse(a_si, a_gt):
    return jnp.mean((a_si - a_gt) ** 2)


def supervised_loss(y_hat, y, a_si=None, a_gt=None, lambda_a: float = 1.0, eps: float = 1e-7):
    """BCE(ŷ, y) plus λ MSE(a_si, a_gt) when affect labels are provided."""
    loss = binary_cross_entropy(y_hat, y, eps=eps)
    if a_si is not None and a_gt is not None:
        loss = loss + lambda_a * affect_mse(a_si, a_gt)
    return loss
