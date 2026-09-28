"""Posterior-mean, DDIM (η=0), and stochastic DDPM reverse samplers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from intrinsic_atlas.schedules import VPSchedule


def posterior_mean_step(x_t, x0_hat, schedule: VPSchedule, t):
    """Appendix A mean update, no re-injected noise.

    x_{t-1} = c_{1,t} x̂0(x_t) + c_{2,t} x_t
    """
    return schedule.c1[t] * x0_hat + schedule.c2[t] * x_t


def ddim_step(x_t, x0_hat, schedule: VPSchedule, t):
    """DDIM η=0: re-insert predicted noise at the schedule amplitude.

    x_{t-1} = √ᾱ_{t-1} x̂0 + √(1-ᾱ_{t-1}) ε̂
    ε̂ = (x_t - √ᾱ_t x̂0) / √(1-ᾱ_t)

    This is not the posterior-mean update: DDIM preserves each initialization.
    """
    alpha_bar = schedule.alphas_cumprod[t]
    alpha_bar_prev = schedule.alphas_cumprod_prev[t]
    eps = (x_t - jnp.sqrt(alpha_bar) * x0_hat) / jnp.sqrt(1.0 - alpha_bar)
    return jnp.sqrt(alpha_bar_prev) * x0_hat + jnp.sqrt(1.0 - alpha_bar_prev) * eps


def stochastic_step(x_t, x0_hat, schedule: VPSchedule, t, noise):
    """Ordinary DDPM reverse: posterior mean plus σ̃_t z, z~N(0,I).

    σ̃_t² = β_t (1-ᾱ_{t-1}) / (1-ᾱ_t)  (true reverse posterior variance).
    """
    mean = posterior_mean_step(x_t, x0_hat, schedule, t)
    return mean + schedule.posterior_std[t] * noise


def sample_posterior_mean(x_T, predict_x0, schedule: VPSchedule):
    """Iterate x_{t-1} ← μ_θ(x_t, t) from t=T down to t=1, no noise.

    The common endpoint Φ_θ(x_T) is the intrinsic atlas (for a unimodal score).
    """

    def body(x, t):
        x0_hat = predict_x0(x, t)
        x_prev = posterior_mean_step(x, x0_hat, schedule, t)
        return x_prev, None

    timesteps = jnp.arange(schedule.betas.shape[0] - 1, -1, -1)
    x0, _ = jax.lax.scan(body, x_T, timesteps)
    return x0


def sample_ddim(x_T, predict_x0, schedule: VPSchedule):
    """Deterministic DDIM (η=0) on the same x̂0 predictor."""

    def body(x, t):
        x0_hat = predict_x0(x, t)
        x_prev = ddim_step(x, x0_hat, schedule, t)
        return x_prev, None

    timesteps = jnp.arange(schedule.betas.shape[0] - 1, -1, -1)
    x0, _ = jax.lax.scan(body, x_T, timesteps)
    return x0


def sample_stochastic(x_T, predict_x0, schedule: VPSchedule, key):
    """Stochastic DDPM reverse (noise re-injected at every t>1)."""

    def body(carry, t):
        x, key = carry
        key, sub = jax.random.split(key)
        x0_hat = predict_x0(x, t)
        noise = jax.random.normal(sub, x.shape, dtype=x.dtype)
        x_prev = stochastic_step(x, x0_hat, schedule, t, noise)
        return (x_prev, key), None

    timesteps = jnp.arange(schedule.betas.shape[0] - 1, -1, -1)
    (x0, _), _ = jax.lax.scan(body, (x_T, key), timesteps)
    return x0


def spread_offset(endpoints, center):
    """Table 2 metrics: RMS spread about the centroid, offset of centroid from ``center``."""
    endpoints = jnp.asarray(endpoints)
    center = jnp.asarray(center)
    centroid = jnp.mean(endpoints, axis=0)
    spread = jnp.sqrt(jnp.mean(jnp.sum((endpoints - centroid) ** 2, axis=-1)))
    offset = jnp.linalg.norm(centroid - center)
    return spread, offset
