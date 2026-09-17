"""Robust loss and influence weight.

The paper (Section III-C, eq. (12)) requires a monotonically increasing,
concave, differentiable Psi whose derivative Psi_dot(z) down-weights
samples far from the current subspace. It cites Higuchi and Eguchi 1998
but does not pick a specific Psi.

This kernel uses the log / Cauchy form

    Psi(z) = log(1 + z / c)
    Psi_dot(z) = 1 / (c + z)

with c > 0. Influence decreases as z increases.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

DEFAULT_C = 1.0


def psi(z: Array, c: float = DEFAULT_C) -> Array:
    """Robust loss Psi(z) = log(1 + z/c)."""
    z = jnp.asarray(z)
    return jnp.log1p(z / c)


def psi_dot(z: Array, c: float = DEFAULT_C) -> Array:
    """Influence function Psi_dot(z) = 1 / (c + z)."""
    z = jnp.asarray(z)
    return 1.0 / (c + z)


def rotation_fraction(z: Array, alpha0: float, c: float = DEFAULT_C) -> Array:
    """Per-sample rotation fraction, paper eq. (12): alpha = alpha0 * Psi_dot(z)."""
    return alpha0 * psi_dot(z, c=c)
