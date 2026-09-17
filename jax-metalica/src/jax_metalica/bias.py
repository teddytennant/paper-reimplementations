"""Well-tempered metadynamics bias on a 1D CV grid (eq 10 / 11 / 13)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class CVGrid(NamedTuple):
    omega: Array
    V: Array


def cv_grid(
    n_points: int = 41,
    omega_min: float = -8.0,
    omega_max: float = 8.0,
) -> CVGrid:
    """Uniform CV lattice. Paper uses 321 points on [-8, 8]; tests use coarser."""
    omega = jnp.linspace(omega_min, omega_max, n_points)
    return CVGrid(omega=omega, V=jnp.zeros((n_points,)))


def evaluate_V(omega: Array, V: Array, x: Array) -> Array:
    """Linear interpolation of the tabulated bias, clipped to the grid."""
    return jnp.interp(x, omega, V)


def gaussian_hill(omega: Array, center: Array, sigma: Array | float) -> Array:
    return jnp.exp(-0.5 * (omega - center) ** 2 / (sigma * sigma))


def deposit_hills(
    omega: Array,
    V: Array,
    samples: Array,
    h: float = 0.1,
    sigma: float = 0.2,
    gamma: float = 2.0,
) -> tuple[Array, Array]:
    """Block WTMetaD deposition (eq 13).

    The iteration height budget ``h`` is split evenly across ``N`` samples.
    Each hill is tempered against ``V + b`` already accumulated in the block.
    Returns ``(V + b, b)``.
    """
    samples = jnp.atleast_1d(jnp.asarray(samples))
    n = samples.shape[0]
    height_each = h / jnp.maximum(n, 1)
    inv_temp = 1.0 / (gamma - 1.0)

    def body(bk: Array, x: Array) -> tuple[Array, Array]:
        v_here = evaluate_V(omega, V + bk, x)
        amp = height_each * jnp.exp(-v_here * inv_temp)
        hill = amp * gaussian_hill(omega, x, sigma)
        return bk + hill, hill

    b0 = jnp.zeros_like(V)
    b, _hills = jax.lax.scan(body, b0, samples)
    return V + b, b
