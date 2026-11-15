"""Rotated/shifted/scaled Wolfe-Quapp potential, eq. (S2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Supplementary Table / eq. (S2)
C1 = 0.969233
C2 = -0.480614
C3 = -2.155285
C4 = 0.184602
C5 = -0.285146
C6 = -1.46487
C7 = 0.480614
C8 = 0.136719
C9 = -3.84471
C10 = 0.969233
C11 = 6.76245
H = 4.0  # barrier height in kBT
KT = 1.0
BETA = 1.0
RADIUS = 0.5
KBIAS = 1500.0
DT = 1e-2
D = 1.0

# Newton minima of U (global min shifted near 0). Lie on y=x within ~0.03.
MIN_A = jnp.array([-1.3009592, -1.3330977])
MIN_B = jnp.array([1.3494982, 1.3187299])


def _energy_one(xy):
    x = xy[0]
    y = xy[1]
    poly = (
        C1 * x**4
        + C2 * x**3 * y
        + C3 * x**2
        + C4 * x**2 * y**2
        + C5 * x
        + C6 * x * y
        + C7 * x * y**3
        + C8 * y
        + C9 * y**2
        + C10 * y**4
        + C11
    )
    return (H / C11) * poly


def energy(x):
    """U'(x,y). Accepts (2,) or (N, 2)."""
    x = jnp.asarray(x)
    if x.ndim == 1:
        return _energy_one(x)
    return jax.vmap(_energy_one)(x)


def grad_energy(x):
    x = jnp.asarray(x)
    if x.ndim == 1:
        return jax.grad(_energy_one)(x)
    return jax.vmap(jax.grad(_energy_one))(x)


def force(x):
    return -grad_energy(x)


def in_A(x, radius: float = RADIUS):
    x = jnp.asarray(x)
    d = jnp.linalg.norm(x - MIN_A, axis=-1)
    return d <= radius


def in_B(x, radius: float = RADIUS):
    x = jnp.asarray(x)
    d = jnp.linalg.norm(x - MIN_B, axis=-1)
    return d <= radius


def state_label(x, radius: float = RADIUS):
    """+1 if in A, -1 if in B, 0 otherwise. B wins if a point is in both."""
    a = in_A(x, radius)
    b = in_B(x, radius)
    return jnp.where(b, -1, jnp.where(a, 1, 0))


def biased_energy(x, xi, xi_hat, kbias: float = KBIAS):
    """U'_ξ̂(x) = U(x) + (kbias/2)(ξ(x) - ξ̂)^2, eq. (10)."""
    return energy(x) + 0.5 * kbias * jnp.square(xi - xi_hat)
