"""Analytic 1D bimodal GMM toy (Appendix F.1, eq 22–25)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_metalica.sde import a_t, nu_t

Array = jax.Array

MU_TOY = (-3.0, 3.0)
SIGMA_TOY = 0.5


class ToyGMM(NamedTuple):
    """Two-component 1D Gaussian mixture.

    weights[0]/weights[1] = 10^η with weights summing to 1 (eq 22).
    """

    mu: Array
    sigma: Array
    weights: Array


def mixture_weights(eta: float | Array) -> Array:
    """c1/c2 = 10^η, c1 + c2 = 1. Mode 2 (μ=+3) is the rare mode when η>0."""
    ratio = jnp.asarray(10.0) ** jnp.asarray(eta)
    c2 = 1.0 / (1.0 + ratio)
    c1 = ratio * c2
    return jnp.stack([c1, c2])


def toy_gmm(eta: float = 0.0) -> ToyGMM:
    return ToyGMM(
        mu=jnp.array(MU_TOY),
        sigma=jnp.asarray(SIGMA_TOY),
        weights=mixture_weights(eta),
    )


def marginal_variance(t: Array | float, gmm: ToyGMM) -> Array:
    """s_t^2 = a_t^2 σ_mix^2 + ν_t^2 (eq 23)."""
    a = a_t(t)
    nu = nu_t(t)
    return a * a * gmm.sigma * gmm.sigma + nu * nu


def _component_log_probs(x: Array, t: Array | float, gmm: ToyGMM) -> Array:
    """log N(x; a_t μ_i, s_t^2) for each component, shape (2,)."""
    a = a_t(t)
    s2 = marginal_variance(t, gmm)
    mean = a * gmm.mu
    return -0.5 * (jnp.log(2.0 * jnp.pi * s2) + (x - mean) ** 2 / s2)


def log_p_t(x: Array, t: Array | float, gmm: ToyGMM) -> Array:
    """log p_t(x) of the noised mixture (eq 23)."""
    log_comp = _component_log_probs(x, t, gmm)
    return jax.nn.logsumexp(jnp.log(gmm.weights) + log_comp)


def responsibilities(x: Array, t: Array | float, gmm: ToyGMM) -> Array:
    """Posterior component responsibilities q_i^t(x) (eq 24)."""
    logits = jnp.log(gmm.weights) + _component_log_probs(x, t, gmm)
    return jax.nn.softmax(logits)


def score(x: Array, t: Array | float, gmm: ToyGMM) -> Array:
    """Exact score ∇_x log p_t(x) (eq 25)."""
    a = a_t(t)
    s2 = marginal_variance(t, gmm)
    q = responsibilities(x, t, gmm)
    return -jnp.sum(q * (x - a * gmm.mu) / s2)


def xhat0(x: Array, t: Array | float, gmm: ToyGMM, a_floor: float = 1e-8) -> Array:
    """Tweedie denoiser x̂_{0|t}(x) = (1/a_t)(x + ν_t^2 ∇ log p_t) (eq 2)."""
    a = jnp.maximum(a_t(t), a_floor)
    nu = nu_t(t)
    return (x + nu * nu * score(x, t, gmm)) / a
