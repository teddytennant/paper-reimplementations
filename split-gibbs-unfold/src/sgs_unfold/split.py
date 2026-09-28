"""Split-Gibbs coupling identities (paper eqs. 1--5)."""

from __future__ import annotations

import jax.numpy as jnp


def coupling_kernel_log(s, s_prime, tau):
    """Unnormalized log of the Gaussian coupling in eq. (1):
    π_τ(s,s') ∝ p_S(s) p_{Y|S}(y|s') exp(-||s'-s||² / (2τ²)).
    Returns -||s'-s||² / (2τ²).
    """
    tau = jnp.asarray(tau)
    return -jnp.sum((s_prime - s) ** 2) / (2.0 * tau**2)


def g_tau_log_density_factor(x, z, tau):
    """Bayes factor in eq. (3): g_τ[μ](·|z) ∝ μ(·) exp(-||·-z||² / (2τ²)).
    Returns -||x-z||² / (2τ²).
    """
    tau = jnp.asarray(tau)
    return -jnp.sum((x - z) ** 2) / (2.0 * tau**2)


def pstep_tau_z(s_prime_prev, tau_i):
    """P-step of eq. (5): sample S_i ~ g_{τ_i}[p_S](· | S'_{i-1}).
    Returns (τ, z) = (τ_i, S'_{i-1}).
    """
    return tau_i, s_prime_prev


def lstep_tau(tau_0, tau_i):
    """L-step coupling of eq. (5): τ = τ_0 τ_i / √(τ_0² - τ_i²)."""
    tau_0 = jnp.asarray(tau_0)
    tau_i = jnp.asarray(tau_i)
    return tau_0 * tau_i / jnp.sqrt(tau_0**2 - tau_i**2)


def lstep_z(s_i, tau_0, tau_i):
    """L-step observation of eq. (5): z = τ_0² S_i / (τ_0² - τ_i²)."""
    tau_0 = jnp.asarray(tau_0)
    tau_i = jnp.asarray(tau_i)
    return (tau_0**2) * s_i / (tau_0**2 - tau_i**2)
