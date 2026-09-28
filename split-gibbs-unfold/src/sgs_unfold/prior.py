"""Closed-form Gaussian prior denoiser (paper P-step, μ = p_S).

For p_S = N(0, σ_p² I), D^P(x, η) = (σ_p² / (σ_p² + η²)) x.
Tweedie score: ∇ log p_{X+ηN}(x) = (D(x,η) - x) / η².
"""

from __future__ import annotations

import jax.numpy as jnp


def gaussian_denoiser(x, eta, prior_var):
    """E[S | S+ηN=x] for S ~ N(0, prior_var I)."""
    eta = jnp.asarray(eta, dtype=x.dtype)
    prior_var = jnp.asarray(prior_var, dtype=x.dtype)
    return (prior_var / (prior_var + eta**2)) * x


def gaussian_score(x, eta, prior_var):
    """Analytic score of the Gaussian perturbation kernel via Tweedie."""
    eta = jnp.asarray(eta, dtype=x.dtype)
    d = gaussian_denoiser(x, eta, prior_var)
    return (d - x) / (eta**2)


def make_prior_denoiser(prior_var):
    """Callable D(x, η) for the ODE."""

    def denoiser(x, eta):
        return gaussian_denoiser(x, eta, prior_var)

    return denoiser
