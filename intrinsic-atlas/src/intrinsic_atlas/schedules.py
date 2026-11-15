"""Variance-preserving DDPM schedule and Appendix A coefficients."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class VPSchedule(NamedTuple):
    """Ho et al. linear VP schedule with posterior-mean coefficients.

    Arrays are 0-indexed over t = 0..T-1, matching paper times t = 1..T
    (t = T is the noisiest step). ``alphas_cumprod_prev[0]`` is ᾱ_0 = 1.
    """

    betas: jnp.ndarray
    alphas: jnp.ndarray
    alphas_cumprod: jnp.ndarray
    alphas_cumprod_prev: jnp.ndarray
    c1: jnp.ndarray
    c2: jnp.ndarray
    posterior_std: jnp.ndarray

    @property
    def T(self) -> int:
        return int(self.betas.shape[0])


def linear_beta_schedule(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Linear β_t of Ho et al. 2020, length T."""
    return jnp.linspace(beta_start, beta_end, T)


def make_vp_schedule(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> VPSchedule:
    """Build α_t, ᾱ_t, and Appendix A (c_{1,t}, c_{2,t}).

    c_{1,t} = √ᾱ_{t-1} β_t / (1 - ᾱ_t)
    c_{2,t} = √α_t (1 - ᾱ_{t-1}) / (1 - ᾱ_t)

    Identity: c_{1,t} + c_{2,t} √ᾱ_t = √ᾱ_{t-1}.
    """
    betas = linear_beta_schedule(T, beta_start, beta_end)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas)
    alphas_cumprod_prev = jnp.concatenate([jnp.ones((1,), dtype=alphas.dtype), alphas_cumprod[:-1]])
    one_minus_ab = 1.0 - alphas_cumprod
    c1 = jnp.sqrt(alphas_cumprod_prev) * betas / one_minus_ab
    c2 = jnp.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / one_minus_ab
    posterior_var = betas * (1.0 - alphas_cumprod_prev) / one_minus_ab
    posterior_std = jnp.sqrt(jnp.clip(posterior_var, 0.0))
    return VPSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        c1=c1,
        c2=c2,
        posterior_std=posterior_std,
    )


def rho_t(schedule: VPSchedule, lam):
    """Contraction factor (eq. 4) along an eigen-direction of variance λ.

    ρ_t(λ) = 1 - β_t / (ᾱ_t λ + 1 - ᾱ_t)
    """
    ab = schedule.alphas_cumprod
    return 1.0 - schedule.betas / (ab * lam + 1.0 - ab)


def rho_product(schedule: VPSchedule, lam):
    """∏_t ρ_t(λ), the leftover clean-image scale after T posterior-mean steps."""
    return jnp.prod(rho_t(schedule, lam))
