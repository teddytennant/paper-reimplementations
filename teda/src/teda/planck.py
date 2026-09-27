"""Planck function, eq (1), and a linear-in-T toy contrast."""

from __future__ import annotations

import jax.numpy as jnp

# CODATA 2018 SI constants.
H_PLANCK = 6.62607015e-34  # J s
C_LIGHT = 2.99792458e8  # m s^{-1}
K_BOLTZMANN = 1.380649e-23  # J K^{-1}

_TWO_HC2 = 2.0 * H_PLANCK * C_LIGHT**2
_HC_OVER_K = H_PLANCK * C_LIGHT / K_BOLTZMANN


def planck_B(wavelength_um, temperature_K):
    """Spectral radiance B(lambda, T) in W m^{-2} sr^{-1} um^{-1}.

    Eq (1) is evaluated in SI (lambda in meters), then converted from
    per-meter to per-micrometer wavelength: B_um = B_m * 1e-6.
    """
    lam_m = jnp.asarray(wavelength_um, dtype=jnp.float64) * 1e-6
    temperature_K = jnp.asarray(temperature_K, dtype=jnp.float64)
    x = _HC_OVER_K / (lam_m * temperature_K)
    b_per_m = _TWO_HC2 / (lam_m**5 * (jnp.exp(x) - 1.0))
    return b_per_m * 1e-6


def linear_in_T(temperature, *, scale=1.0, like=None):
    """Toy radiance linear in temperature, avoiding Planck constants in tests."""
    value = jnp.asarray(scale, dtype=jnp.float64) * jnp.asarray(
        temperature, dtype=jnp.float64
    )
    if like is None:
        return value
    return jnp.full_like(jnp.asarray(like, dtype=jnp.float64), value)
