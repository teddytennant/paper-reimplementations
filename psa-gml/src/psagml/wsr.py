"""SINR (eq 8) and weighted sum-rate (eq 9)."""

from __future__ import annotations

import jax.numpy as jnp


def sinr(A, W, sigma2):
    """gamma_k = |a_k^H w_k|^2 / (sum_{j != k} |a_k^H w_j|^2 + sigma^2).

    Rows of A are a_k^H, so (A W)_{kj} = a_k^H w_j.
    """
    aw = A @ W
    signal = jnp.abs(jnp.diag(aw)) ** 2
    total = jnp.sum(jnp.abs(aw) ** 2, axis=1)
    return signal / (total - signal + sigma2)


def weighted_sum_rate(A, W, omega, sigma2):
    """R = sum_k omega_k log2(1 + gamma_k)."""
    gamma = sinr(A, W, sigma2)
    return jnp.sum(omega * jnp.log2(1.0 + gamma))
