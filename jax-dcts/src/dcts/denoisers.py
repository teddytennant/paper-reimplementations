"""Closed-form denoisers. No pretrained image models.

The paper's plug-in is D(x, sigma) ≈ E[Z | X = x] under the VE kernel
X = Z + sigma eps. Tests use an isotropic Gaussian prior, optionally a
two-component Gaussian mixture.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

Denoiser = Callable[[Array, Array], Array]


def gaussian_ve_denoiser(prior_var: float = 1.0) -> Denoiser:
    """MMSE denoiser for Z ~ N(0, prior_var I) and X = Z + sigma eps."""

    def denoise(x: Array, sigma: Array) -> Array:
        shrink = prior_var / (prior_var + sigma**2)
        return shrink * x

    return denoise


def gaussian_vp_denoiser(prior_var: float = 1.0) -> Denoiser:
    """MMSE denoiser for Z ~ N(0, prior_var I) and X = alpha Z + sigma eps.

    Under VP, alpha^2 + sigma^2 = 1, so this is
        E[Z|X] = (prior_var alpha / (prior_var alpha^2 + sigma^2)) X.
    The returned callable still has the paper signature D(x, sigma),
    recovering alpha = sqrt(1 - sigma^2).
    """

    def denoise(x: Array, sigma: Array) -> Array:
        alpha = jnp.sqrt(jnp.maximum(1.0 - sigma**2, 0.0))
        weight = (prior_var * alpha) / (prior_var * alpha**2 + sigma**2)
        return weight * x

    return denoise


def mixture_ve_denoiser(
    means: Array,
    prior_var: float = 1.0,
    weights: Array | None = None,
) -> Denoiser:
    """MMSE denoiser for an equal-weight (default) isotropic Gaussian mixture.

    Z ~ sum_k w_k N(mu_k, prior_var I), X = Z + sigma eps.
    """
    means = jnp.asarray(means)
    k = means.shape[0]
    if weights is None:
        weights = jnp.ones((k,), dtype=means.dtype) / k
    else:
        weights = jnp.asarray(weights)

    def denoise(x: Array, sigma: Array) -> Array:
        var = prior_var + sigma**2
        shrink = prior_var / var
        # log N(x; mu, var I) up to a common constant
        log_comp = -0.5 * jnp.sum((x[None, :] - means) ** 2, axis=-1) / var
        log_comp = log_comp + jnp.log(weights)
        log_comp = log_comp - jnp.max(log_comp)
        resp = jnp.exp(log_comp)
        resp = resp / jnp.sum(resp)
        post_means = shrink * x[None, :] + (1.0 - shrink) * means
        return jnp.sum(resp[:, None] * post_means, axis=0)

    return denoise


def oracle_denoiser(z: Array) -> Denoiser:
    """Returns the known clean signal. Used to test exact Euler transport."""

    z = jnp.asarray(z)

    def denoise(x: Array, sigma: Array) -> Array:
        return jnp.broadcast_to(z, x.shape)

    return denoise
