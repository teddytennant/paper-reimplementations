"""Gaussian ISRF integration, eq (8) and eq (10)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def fwhm_to_sigma(fwhm_um):
    """Gaussian sigma from FWHM: FWHM = 2 sqrt(2 ln 2) sigma."""
    return jnp.asarray(fwhm_um, dtype=jnp.float64) / (
        2.0 * jnp.sqrt(2.0 * jnp.log(2.0))
    )


def trapz(y, x):
    """Trapezoidal integral along the last axis of y versus 1-D x."""
    dx = x[1:] - x[:-1]
    return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, axis=-1)


def gaussian_isrf_weights(lam_fine, lam_center, fwhm_um):
    """Normalized discrete Gaussian ISRF f_k(lambda) on a fine wavelength grid."""
    sigma = fwhm_to_sigma(fwhm_um)
    w = jnp.exp(-0.5 * ((lam_fine - lam_center) / sigma) ** 2)
    norm = trapz(w, lam_fine)
    return w / norm


def isrf_apply(lam_fine, values_fine, lam_channels, fwhm_um):
    """Hk[g] = integral f_k(lambda) g(lambda) d lambda, eq (8).

    ``values_fine`` is g on ``lam_fine``. Returns one value per channel.
    """
    lam_fine = jnp.asarray(lam_fine, dtype=jnp.float64)
    values_fine = jnp.asarray(values_fine, dtype=jnp.float64)
    lam_channels = jnp.asarray(lam_channels, dtype=jnp.float64)

    def one_channel(lam_k):
        w = gaussian_isrf_weights(lam_fine, lam_k, fwhm_um)
        return trapz(w * values_fine, lam_fine)

    return jax.vmap(one_channel)(lam_channels)
