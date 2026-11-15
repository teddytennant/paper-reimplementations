"""Synthetic Lorentzian absorption and toy LWIR observations. No HITRAN."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from teda.isrf import isrf_apply

# Default LWIR window, 8-13 um.
LAM_MIN_UM = 8.0
LAM_MAX_UM = 13.0

# Center (um), HWHM gamma (um), peak amplitude (1/m).
_DEFAULT_LINES = (
    (8.23, 0.018, 0.010),
    (8.52, 0.014, 0.008),
    (9.35, 0.022, 0.004),
    (11.70, 0.016, 0.009),
    (12.40, 0.012, 0.012),
    (12.85, 0.015, 0.007),
)


def lorentzian_line(lam_um, center_um, gamma_um, amplitude):
    """Lorentzian absorption line: A * gamma^2 / ((lam - center)^2 + gamma^2)."""
    lam_um = jnp.asarray(lam_um, dtype=jnp.float64)
    dc = lam_um - jnp.asarray(center_um, dtype=jnp.float64)
    g = jnp.asarray(gamma_um, dtype=jnp.float64)
    return jnp.asarray(amplitude, dtype=jnp.float64) * (g * g) / (dc * dc + g * g)


def synthetic_kappa(lam_um, lines=_DEFAULT_LINES, floor=1e-5):
    """Sum of a few Lorentzian lines on the LWIR window, 1/m."""
    lam_um = jnp.asarray(lam_um, dtype=jnp.float64)
    kappa = jnp.full_like(lam_um, jnp.asarray(floor, dtype=jnp.float64))
    for center, gamma, amp in lines:
        kappa = kappa + lorentzian_line(lam_um, center, gamma, amp)
    return kappa


def channel_grid(n_channels: int = 64):
    return jnp.linspace(LAM_MIN_UM, LAM_MAX_UM, n_channels, dtype=jnp.float64)


def fine_grid(n_fine: int = 1001, pad_um: float = 0.2):
    return jnp.linspace(
        LAM_MIN_UM - pad_um, LAM_MAX_UM + pad_um, n_fine, dtype=jnp.float64
    )


def beer_lambert(kappa, distance_m):
    """tau(lambda; d) = exp(-kappa(lambda) d), eq (7). No ISRF."""
    return jnp.exp(-jnp.asarray(kappa, dtype=jnp.float64) * jnp.asarray(distance_m, dtype=jnp.float64))


def channel_kappa(lam_fine, kappa_fine, lam_channels, fwhm_um=None):
    """Project fine-grid kappa onto channels: interp, or H_k[kappa] if ISRF."""
    if fwhm_um is None:
        return jnp.interp(lam_channels, lam_fine, kappa_fine)
    return isrf_apply(lam_fine, kappa_fine, lam_channels, fwhm_um)


def sensor_transmittance(distance_m, lam_fine, kappa_fine, lam_channels, fwhm_um=None):
    """Channel transmittance: Beer-Lambert on the channel grid, or eq (10) with ISRF."""
    if fwhm_um is None:
        kappa_ch = jnp.interp(lam_channels, lam_fine, kappa_fine)
        return beer_lambert(kappa_ch, distance_m)
    tau_fine = beer_lambert(kappa_fine, distance_m)
    return isrf_apply(lam_fine, tau_fine, lam_channels, fwhm_um)


def transmittance_bank(d_grid, lam_fine, kappa_fine, lam_channels, fwhm_um=None):
    """Sensor-domain tau(d) for every candidate distance. vmap over d, not pixels."""

    def _one(d):
        return sensor_transmittance(d, lam_fine, kappa_fine, lam_channels, fwhm_um)

    return jax.vmap(_one)(jnp.asarray(d_grid, dtype=jnp.float64))


def observe_radiance(tau, B_obj, B_air, emissivity=1.0):
    """y_k from eq (6): tau * eps * B(T) + (1 - tau) * B(T_air)."""
    tau = jnp.asarray(tau, dtype=jnp.float64)
    B_obj = jnp.asarray(B_obj, dtype=jnp.float64)
    B_air = jnp.asarray(B_air, dtype=jnp.float64)
    eps = jnp.asarray(emissivity, dtype=jnp.float64)
    return tau * eps * B_obj + (1.0 - tau) * B_air
