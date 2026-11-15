"""Value encoding v(s) (paper Sec. 3.2)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from mdl.constants import D, SIGMA

jax.config.update("jax_enable_x64", True)


def rbf_centers(d: int = D) -> jnp.ndarray:
    """Equally spaced centers c_i = (i-1)/(D-1) for i = 1..D."""
    return jnp.arange(d, dtype=jnp.float64) / (d - 1)


def value_encode(s: float, sigma: float = SIGMA, d: int = D) -> jnp.ndarray:
    """Map a scalar s in [0, 1] to a D-vector.

    v(s) = s * 1_D + sum_i exp( -(s - c_i)^2 / (2 sigma^2) ) e_i
    """
    s = jnp.asarray(s, dtype=jnp.float64)
    centers = rbf_centers(d)
    rbf = jnp.exp(-((s - centers) ** 2) / (2.0 * sigma**2))
    return s + rbf
