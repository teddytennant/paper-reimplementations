"""Reverse-time denoising ODE, paper eq. (6).

dX̄_t / dt = (σ̇(t)/σ(t)) ( X̄_t - D^μ_θ( (τ² X̄_t + σ(t)² z)/(τ²+σ(t)²),
                                         τ σ(t)/√(τ²+σ(t)²) ) )
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def denoise_argument(xbar, z, tau, sigma):
    """First denoiser argument in eq. (6): (τ² X̄ + σ² z) / (τ² + σ²)."""
    tau2 = tau**2
    sig2 = sigma**2
    return (tau2 * xbar + sig2 * z) / (tau2 + sig2)


def denoise_noise_level(tau, sigma):
    """Second denoiser argument in eq. (6): τ σ / √(τ² + σ²)."""
    return tau * sigma / jnp.sqrt(tau**2 + sigma**2)


def ode_rhs(xbar, z, tau, sigma, sigma_dot_over_sigma, denoiser):
    """Right-hand side of eq. (6). `denoiser(x, eta)` ≈ E[X | X+ηN=x]."""
    arg = denoise_argument(xbar, z, tau, sigma)
    eta = denoise_noise_level(tau, sigma)
    d = denoiser(arg, eta)
    return sigma_dot_over_sigma * (xbar - d)


def euler_reverse_step(xbar, z, tau, sigma, sigma_dot_over_sigma, dt, denoiser):
    """One Euler step of eq. (6). Reverse-time uses dt < 0."""
    return xbar + dt * ode_rhs(
        xbar, z, tau, sigma, sigma_dot_over_sigma, denoiser
    )


def simulate_reverse_ode(
    key,
    z,
    tau,
    denoiser,
    t_grid,
    sigma_fn,
    sigma_dot_over_sigma_fn,
):
    """Initialize X̄_{t0} ~ N(0, σ(t0)² I) and integrate eq. (6) along t_grid.

    `t_grid` must be decreasing from T down toward 0.
    """
    t0 = t_grid[0]
    x = sigma_fn(t0) * jax.random.normal(key, z.shape, dtype=z.dtype)

    def body(x, pair):
        t, t_next = pair
        dt = t_next - t
        sigma = sigma_fn(t)
        sdos = sigma_dot_over_sigma_fn(t)
        x = euler_reverse_step(x, z, tau, sigma, sdos, dt, denoiser)
        return x, None

    x, _ = jax.lax.scan(body, x, (t_grid[:-1], t_grid[1:]))
    return x
