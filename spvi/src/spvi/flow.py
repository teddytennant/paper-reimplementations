"""Closed-form decay flow between events (eq. 1)."""

import jax.nn as jnn
import jax.numpy as jnp


def positive_gamma(raw):
    """Map an unconstrained parameter to gamma > 0, elementwise."""
    return jnn.softplus(raw)


def flow(z, delta, gamma):
    """Phi_delta(z) = exp(-gamma * delta) * z, elementwise in the state.

    z: (..., d)
    delta: broadcastable to z's leading shape (no trailing state axis)
    gamma: (d,) or broadcastable to z
    """
    z = jnp.asarray(z)
    delta = jnp.asarray(delta)
    gamma = jnp.asarray(gamma)
    return jnp.exp(-gamma * delta[..., None]) * z
