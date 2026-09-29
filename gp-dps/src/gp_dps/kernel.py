"""SE-ARD kernel and first-argument input derivatives (paper §3)."""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def se_ard(x: Array, x2: Array, ell: Array, eta: Array) -> Array:
    """k(x, x') = eta * exp(-0.5 * sum_d (x_d - x'_d)^2 / ell_d^2).

    x: (n, D), x2: (m, D), ell: (D,), eta: scalar. Returns (n, m).
    """
    diff = x[:, None, :] - x2[None, :, :]
    quad = jnp.sum((diff ** 2) / (ell ** 2), axis=-1)
    return eta * jnp.exp(-0.5 * quad)


def se_ard_deriv_first(x: Array, x2: Array, ell: Array, eta: Array, j: int | Array) -> Array:
    """∂k(x, x') / ∂x_j = -(x_j - x'_j) / ell_j^2 * k(x, x').

    Derivative is with respect to coordinate j of the first argument.
    Returns (n, m).
    """
    k = se_ard(x, x2, ell, eta)
    dxj = x[:, j][:, None] - x2[:, j][None, :]
    return -(dxj / (ell[j] ** 2)) * k


def prior_deriv_variance(ell: Array, eta: Array) -> Array:
    """Prior derivative variance on the diagonal: eta / ell_j^2 for each j."""
    return eta / (ell ** 2)
