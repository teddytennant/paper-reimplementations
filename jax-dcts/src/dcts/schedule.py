"""Noise schedules for the DCTS kernel.

The paper writes the general Gaussian kernel
    q_t(x | z) = N(x; alpha_t z, sigma_t^2 I)
and then adopts VE, where alpha_t ≡ 1. VP is the same kernel with
alpha_t^2 + sigma_t^2 = 1. Algorithm 1 uses VE outer levels from the
EDM discretization plus a final zero.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def vp_alpha_sigma(t: Array) -> tuple[Array, Array]:
    """Cosine VP schedule, t=0 clean, t=1 fully noisy.

    alpha(t) = cos(pi t / 2), sigma(t) = sin(pi t / 2), so
    alpha^2 + sigma^2 = 1.
    """
    t = jnp.asarray(t)
    half_pi = 0.5 * jnp.pi
    alpha = jnp.cos(half_pi * t)
    sigma = jnp.sin(half_pi * t)
    return alpha, sigma


def vp_bar_alpha(t: Array) -> Array:
    """DDPM-style bar-alpha: alpha_t^2 under the VP identity."""
    alpha, _sigma = vp_alpha_sigma(t)
    return alpha**2


def vp_discrete(num_steps: int, t_min: float = 0.0, t_max: float = 1.0) -> tuple[Array, Array]:
    """Decreasing-noise VP grid: first index is noisiest.

    Returns (alpha, sigma) each of shape (num_steps + 1,), matching
    Algorithm 1's sigma_0 > ... > sigma_L, with L = num_steps.
    """
    # tau runs from t_max (noisy) down to t_min (clean)
    tau = jnp.linspace(t_max, t_min, num_steps + 1)
    return vp_alpha_sigma(tau)


def ve_alpha(sigma: Array) -> Array:
    """VE parameterization: alpha_t ≡ 1."""
    return jnp.ones_like(jnp.asarray(sigma))


def edm_sigmas(
    num_outer: int,
    sigma_max: float = 100.0,
    sigma_min: float = 0.1,
    rho: float = 7.0,
    append_zero: bool = True,
    dtype=jnp.float32,
) -> Array:
    """EDM polynomial outer schedule used in the paper.

    sigma_i = (sigma_max^{1/rho} + i/(N-1) (sigma_min^{1/rho} - sigma_max^{1/rho}))^rho
    for i = 0, ..., N-1 with N = num_outer, then a final 0 if append_zero.

    With append_zero=True this yields L = num_outer transitions and
    sigma_0 > ... > sigma_L = 0 as in Algorithm 1.
    """
    if num_outer < 1:
        raise ValueError("num_outer must be >= 1")
    n = num_outer
    i = jnp.arange(n, dtype=dtype)
    inv_rho = 1.0 / rho
    start = sigma_max**inv_rho
    stop = sigma_min**inv_rho
    if n == 1:
        sigmas = jnp.array([sigma_max], dtype=dtype)
    else:
        sigmas = (start + (i / (n - 1)) * (stop - start)) ** rho
    if append_zero:
        sigmas = jnp.concatenate([sigmas, jnp.zeros((1,), dtype=dtype)])
    return sigmas
