"""Functional calibration objective (Eq. 5) and path functionals."""

from __future__ import annotations

import jax.numpy as jnp

from .girsanov import snis_expect


def calibration_loss(estimates, targets, weights=None):
    """L_n(θ) = sum_φ w_φ (Ê[φ] - c_φ)^2  (Eq. 5)."""
    estimates = jnp.stack(estimates)
    targets = jnp.stack(targets)
    err = estimates - targets
    if weights is None:
        return jnp.mean(err * err)
    weights = jnp.asarray(weights)
    return jnp.sum(weights * err * err) / jnp.sum(weights)


def mc_means(values):
    """Plain Monte-Carlo Ê[φ] = n^{-1} sum_i φ(Y^i)."""
    return jnp.mean(values, axis=0)


def weighted_means(values, log_dp_dq=None):
    """MC mean, or SNIS mean when log dP/dQ is supplied."""
    if log_dp_dq is None:
        return mc_means(values)
    return snis_expect(values, log_dp_dq)


def terminal(y):
    """φ(Y) = Y_T  (first observation channel)."""
    return y[:, -1, 0]


def path_mean(y):
    """φ(Y) = N^{-1} sum_k Y_{t_k}."""
    return jnp.mean(y[:, :, 0], axis=1)


def path_mean_square(y):
    """φ(Y) = N^{-1} sum_k Y_{t_k}^2  (squared path average)."""
    return jnp.mean(y[:, :, 0] ** 2, axis=1)


def running_max(y):
    """φ(Y) = max_k Y_{t_k}."""
    return jnp.max(y[:, :, 0], axis=1)


def positive_part_terminal(y):
    """φ(Y) = (Y_T)_+."""
    return jnp.maximum(y[:, -1, 0], 0.0)
