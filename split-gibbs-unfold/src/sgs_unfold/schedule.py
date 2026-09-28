"""Noise-level schedules for paper eq. (6).

Paper eq. (6) takes a user-specified increasing σ(t). The VP identity
σ²(t) = exp(∫_0^t β_s ds) - 1, with inverse
t = inf{u : ∫_0^u β = log(1+σ²)}, is one such choice.
"""

from __future__ import annotations

import jax.numpy as jnp


def integrate_beta(beta, t):
    """∫_0^t β_s ds for constant β > 0."""
    return jnp.asarray(beta, dtype=t.dtype) * t


def vp_sigma_sq(t, beta):
    """σ²(t) = exp(∫_0^t β_s ds) - 1."""
    t = jnp.asarray(t)
    return jnp.exp(integrate_beta(beta, t)) - 1.0


def vp_t_from_sigma_sq(sigma_sq, beta):
    """t = inf{u : ∫_0^u β = log(1+σ²)} for constant β > 0."""
    sigma_sq = jnp.asarray(sigma_sq)
    beta = jnp.asarray(beta, dtype=sigma_sq.dtype)
    return jnp.log1p(sigma_sq) / beta


def vp_sigma(t, beta):
    """σ(t) = sqrt(exp(∫_0^t β) - 1)."""
    return jnp.sqrt(vp_sigma_sq(t, beta))


def vp_sigma_dot_over_sigma(t, beta):
    """σ̇(t)/σ(t) for the VP schedule. σ̇/σ = β (σ²+1) / (2 σ²)."""
    t = jnp.asarray(t)
    beta = jnp.asarray(beta, dtype=t.dtype)
    s2 = vp_sigma_sq(t, beta)
    return beta * (s2 + 1.0) / (2.0 * s2)


def edm_sigma(t):
    """σ(t) = t (EDM / Karras convention)."""
    return jnp.asarray(t)


def edm_sigma_dot_over_sigma(t):
    """σ̇/σ = 1/t when σ(t)=t."""
    return 1.0 / jnp.asarray(t)
