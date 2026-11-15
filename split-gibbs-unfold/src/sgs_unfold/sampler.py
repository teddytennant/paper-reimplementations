"""Alternating P-step / L-step ODE sampler (paper Fig. 1, eq. 5).

Not a joint MCMC chain: the prior update never sees y; the likelihood
update never uses the prior density, only D^L(·,·; y).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sgs_unfold.ode import simulate_reverse_ode
from sgs_unfold.split import lstep_tau, lstep_z, pstep_tau_z


def decreasing_t_grid(t_max, t_min, n_steps):
    return jnp.linspace(t_max, t_min, n_steps + 1)


def p_step(
    key,
    s_prime_prev,
    tau_i,
    prior_denoiser,
    t_grid,
    sigma_fn,
    sigma_dot_over_sigma_fn,
):
    """Prior update: sample S_i ~ g_{τ_i}[p_S](· | S'_{i-1}) via eq. (6).

    Does not take the observation y.
    """
    tau, z = pstep_tau_z(s_prime_prev, tau_i)
    return simulate_reverse_ode(
        key, z, tau, prior_denoiser, t_grid, sigma_fn, sigma_dot_over_sigma_fn
    )


def l_step(
    key,
    s_i,
    tau_i,
    tau_0,
    likelihood_denoiser,
    t_grid,
    sigma_fn,
    sigma_dot_over_sigma_fn,
):
    """Likelihood update: sample S'_i via eq. (5) L-step and eq. (6).

    `likelihood_denoiser(x, η)` already closes over y.
    """
    tau = lstep_tau(tau_0, tau_i)
    z = lstep_z(s_i, tau_0, tau_i)
    return simulate_reverse_ode(
        key,
        z,
        tau,
        likelihood_denoiser,
        t_grid,
        sigma_fn,
        sigma_dot_over_sigma_fn,
    )


def split_gibbs_step(
    key,
    s_prime_prev,
    tau_i,
    tau_0,
    prior_denoiser,
    likelihood_denoiser,
    t_grid,
    sigma_fn,
    sigma_dot_over_sigma_fn,
):
    """One outer iteration: P-step then L-step. Returns (S_i, S'_i)."""
    k_p, k_l = jax.random.split(key)
    s_i = p_step(
        k_p,
        s_prime_prev,
        tau_i,
        prior_denoiser,
        t_grid,
        sigma_fn,
        sigma_dot_over_sigma_fn,
    )
    s_prime = l_step(
        k_l,
        s_i,
        tau_i,
        tau_0,
        likelihood_denoiser,
        t_grid,
        sigma_fn,
        sigma_dot_over_sigma_fn,
    )
    return s_i, s_prime


def tau_schedule(tau_max, tau_min, n_outer):
    """Decreasing coupling strengths with τ_i < τ_0 (caller enforces)."""
    if n_outer == 1:
        return jnp.array([tau_max])
    return jnp.exp(jnp.linspace(jnp.log(tau_max), jnp.log(tau_min), n_outer))


def split_gibbs_sample(
    key,
    s_prime0,
    tau_0,
    taus,
    prior_denoiser,
    likelihood_denoiser,
    t_grid,
    sigma_fn,
    sigma_dot_over_sigma_fn,
):
    """K outer alternating denoising steps. Returns (S_K, S'_K)."""
    s_prime = s_prime0
    s = s_prime0
    for tau_i in taus:
        key, k_step = jax.random.split(key)
        s, s_prime = split_gibbs_step(
            k_step,
            s_prime,
            tau_i,
            tau_0,
            prior_denoiser,
            likelihood_denoiser,
            t_grid,
            sigma_fn,
            sigma_dot_over_sigma_fn,
        )
    return s, s_prime
