"""Euler–Maruyama kernels and the closed-form backward ratio (Appendix E)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_metalica.bias import evaluate_V
from jax_metalica.gmm import ToyGMM, log_p_t, score, xhat0
from jax_metalica.sde import beta, guidance_lambda

Array = jax.Array


def log_normal(x: Array, mean: Array, var: Array) -> Array:
    """log N(x; mean, var) for scalar (or broadcast) 1D Gaussians."""
    var = jnp.maximum(var, 1e-12)
    return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (x - mean) ** 2 / var)


def forward_mean(x: Array, t_lo: Array | float, dt: Array | float) -> Array:
    """Upward EM mean: x - ½ β_lo x Δt (eq 20)."""
    return x - 0.5 * beta(t_lo) * x * dt


def forward_var(t_lo: Array | float, dt: Array | float) -> Array:
    """Upward EM variance: β_lo Δt."""
    return beta(t_lo) * dt


def pretrained_backward_mean(
    y: Array, t_hi: Array | float, dt: Array | float, gmm: ToyGMM
) -> Array:
    """μ_P = y + (½ β_hi y + β_hi score(y, t_m)) Δt (eq 20)."""
    bhi = beta(t_hi)
    s = score(y, t_hi, gmm)
    return y + (0.5 * bhi * y + bhi * s) * dt


def reward(
    x: Array,
    t: Array | float,
    omega: Array,
    V: Array,
    gmm: ToyGMM,
    rho: float = 0.5,
) -> Array:
    """r_t(x) = -λ_t V(ξ(x̂_{0|t}(x))) with ξ(x)=x (eq 4 / 9)."""
    lam = guidance_lambda(t, rho=rho)
    x0 = xhat0(x, t, gmm)
    return -lam * evaluate_V(omega, V, x0)


def grad_reward(
    x: Array,
    t: Array | float,
    omega: Array,
    V: Array,
    gmm: ToyGMM,
    rho: float = 0.5,
) -> Array:
    """∇_x r_t(x) via reverse-mode autodiff through Tweedie and V(ω)."""

    def _r(z: Array) -> Array:
        return reward(z, t, omega, V, gmm, rho=rho)

    return jax.grad(_r)(jnp.asarray(x))


def tilted_backward_mean(
    y: Array,
    t_hi: Array | float,
    dt: Array | float,
    omega: Array,
    V: Array,
    gmm: ToyGMM,
    rho: float = 0.5,
) -> Array:
    """μ_Q = μ_P + β_hi ∇r_{t_m}(y) Δt (eq 20)."""
    mu_p = pretrained_backward_mean(y, t_hi, dt, gmm)
    gr = grad_reward(y, t_hi, omega, V, gmm, rho=rho)
    return mu_p + beta(t_hi) * gr * dt


def log_B0_minus_log_B(
    x: Array,
    y: Array,
    t_hi: Array | float,
    dt: Array | float,
    omega: Array,
    V: Array,
    gmm: ToyGMM,
    rho: float = 0.5,
) -> Array:
    """Closed form (eq 21): -(x-μ_P)·∇r + (v/2)||∇r||^2, v = β_hi Δt."""
    mu_p = pretrained_backward_mean(y, t_hi, dt, gmm)
    gr = grad_reward(y, t_hi, omega, V, gmm, rho=rho)
    v = beta(t_hi) * dt
    return -(x - mu_p) * gr + 0.5 * v * (gr * gr)


def log_pi(
    x: Array,
    t: Array | float,
    omega: Array,
    V: Array,
    gmm: ToyGMM,
    rho: float = 0.5,
) -> Array:
    """log π_t(x) = log p_t(x) - λ_t V(ξ(x̂_{0|t}(x))) up to a t-only constant."""
    return log_p_t(x, t, gmm) + reward(x, t, omega, V, gmm, rho=rho)


def incremental_log_weight(
    x_lo: Array,
    x_hi: Array,
    t_lo: Array | float,
    t_hi: Array | float,
    omega: Array,
    V: Array,
    gmm: ToyGMM,
    rho: float = 0.5,
) -> Array:
    """log g_{m-1|m} of eq 14, shared by sequential (SMC) and parallel control.

    Omits the ratio of π-normalizers, which cancels in normalized weights and
    in the swap ratio. Thin MetaDiff/SMC hook: one backward kernel evaluation.
    """
    dt = t_hi - t_lo
    r_lo = reward(x_lo, t_lo, omega, V, gmm, rho=rho)
    r_hi = reward(x_hi, t_hi, omega, V, gmm, rho=rho)
    log_F = log_normal(x_hi, forward_mean(x_lo, t_lo, dt), forward_var(t_lo, dt))
    mu_p = pretrained_backward_mean(x_hi, t_hi, dt, gmm)
    v_hi = beta(t_hi) * dt
    log_B0 = log_normal(x_lo, mu_p, v_hi)
    mu_q = mu_p + v_hi * grad_reward(x_hi, t_hi, omega, V, gmm, rho=rho)
    log_B = log_normal(x_lo, mu_q, v_hi)
    log_p_ratio = log_p_t(x_lo, t_lo, gmm) - log_p_t(x_hi, t_hi, gmm)
    return (r_lo - r_hi) + log_p_ratio + log_F - log_B0 + (log_B0 - log_B)
