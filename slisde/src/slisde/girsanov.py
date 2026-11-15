"""Last-layer Girsanov tilt with exact likelihood-ratio weights (Eqs. 10–12, 76, 83)."""

from __future__ import annotations

import jax.numpy as jnp


def correlated_driver(dW_prefix, d_eps, rho):
    """dW^{(L)} = ρ dW^{prefix} + sqrt(1-ρ^2) dε  (Eq. 10)."""
    rho = jnp.asarray(rho)
    return rho * dW_prefix + jnp.sqrt(jnp.clip(1.0 - rho * rho, 0.0, 1.0)) * d_eps


def log_dq_over_dp(u, d_eps_P, dt, mu=None, z0=None):
    """log dQ/dP  (Eq. 12, discrete Eq. 76).

    log dQ/dP = μ^T z0 - 1/2 ||μ||^2 + sum_k (u_k^T Δε^P_k - 1/2 ||u_k||^2 Δt)
    """
    step = jnp.sum(u * d_eps_P, axis=-1) - 0.5 * jnp.sum(u * u, axis=-1) * dt
    log_r = jnp.sum(step, axis=-1)
    if mu is not None:
        mu = jnp.asarray(mu)
        z0 = jnp.zeros_like(mu) if z0 is None else z0
        log_r = log_r + jnp.sum(mu * z0, axis=-1) - 0.5 * jnp.sum(mu * mu, axis=-1)
    return log_r


def log_dp_over_dq(u, d_eps_Q, dt, mu=None, z0=None):
    """log dP/dQ  (Eq. 76), the SNIS weight on paths simulated under Q.

    log dP/dQ = -μ^T z0 + 1/2 ||μ||^2 - sum_k (u_k^T Δε^Q_k + 1/2 ||u_k||^2 Δt)
    """
    step = jnp.sum(u * d_eps_Q, axis=-1) + 0.5 * jnp.sum(u * u, axis=-1) * dt
    log_r = -jnp.sum(step, axis=-1)
    if mu is not None:
        mu = jnp.asarray(mu)
        z0 = jnp.zeros_like(mu) if z0 is None else z0
        log_r = log_r - jnp.sum(mu * z0, axis=-1) + 0.5 * jnp.sum(mu * mu, axis=-1)
    return log_r


def _finite_log(log_w):
    """Keep likelihood-ratio logs finite (no NaNs)."""
    return jnp.nan_to_num(log_w, nan=0.0, posinf=80.0, neginf=-80.0)


def snis_weights(log_dp_dq):
    """Self-normalised importance weights (Eq. 83), numerically stable."""
    log_w = _finite_log(log_dp_dq)
    centered = log_w - jnp.max(log_w)
    w = jnp.exp(centered)
    return w / jnp.sum(w)


def snis_expect(values, log_dp_dq):
    """SNIS estimate of E_P[φ] from Q-samples (Eq. 83)."""
    w = snis_weights(log_dp_dq)
    return jnp.sum(w * values, axis=0)


def ess(log_dp_dq):
    """Effective sample size (Eq. 84)."""
    w = snis_weights(log_dp_dq)
    return 1.0 / jnp.sum(w * w)


def tilt_innovation(d_eps_Q, u, dt):
    """Δε^P = Δε^Q + u Δt, the physical innovation under a Q-simulation."""
    return d_eps_Q + u * dt
