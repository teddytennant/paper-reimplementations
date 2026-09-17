"""Cosine VP-SDE schedule (Appendix E / F.1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array

COSINE_DELTA = 0.008
# Clamp so tan(π/2 · (t+δ)/(1+δ)) stays finite at t → 1.
BETA_T_MAX = 1.0 - 1e-3


def a_t(t: Array | float, delta: float = COSINE_DELTA) -> Array:
    """Signal coefficient a_t of the cosine VP schedule (Appendix F.1)."""
    t = jnp.asarray(t)
    inner = (jnp.pi / 2.0) * (t + delta) / (1.0 + delta)
    norm = jnp.cos((jnp.pi / 2.0) * delta / (1.0 + delta))
    return jnp.cos(inner) / norm


def nu_t(t: Array | float, delta: float = COSINE_DELTA) -> Array:
    """Noise level ν_t = sqrt(max(1 - a_t^2, 0))."""
    a = a_t(t, delta=delta)
    return jnp.sqrt(jnp.maximum(1.0 - a * a, 0.0))


def beta(t: Array | float, delta: float = COSINE_DELTA, t_max: float = BETA_T_MAX) -> Array:
    """Instantaneous VP diffusion coefficient β(t) = -2 d(log a_t)/dt.

    Closed form from a_t = cos(θ(t))/const with θ = (π/2)(t+δ)/(1+δ):
        β(t) = (π/(1+δ)) tan(θ(t)).
    """
    t = jnp.clip(jnp.asarray(t), 0.0, t_max)
    theta = (jnp.pi / 2.0) * (t + delta) / (1.0 + delta)
    return (jnp.pi / (1.0 + delta)) * jnp.tan(theta)


def guidance_lambda(t: Array | float, rho: float = 0.5) -> Array:
    """Guidance schedule λ_t = (1-t)^ρ (eq 19). ρ=0.5 on the toy."""
    t = jnp.asarray(t)
    return jnp.maximum(1.0 - t, 0.0) ** rho


def time_grid(M: int) -> Array:
    """Uniform schedule 0 = t_0 < … < t_M = 1 (Appendix B)."""
    return jnp.linspace(0.0, 1.0, M + 1)


alpha = a_t
lambda_t = guidance_lambda
