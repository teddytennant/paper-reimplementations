"""Relative RMSE used in the ADE experiments."""

import jax.numpy as jnp


def rrmse(u, u_ref, u0=1.0):
    """rRMSE_u = ||u - u_ref||_2 / (u0 √N_u)."""
    u = jnp.asarray(u).reshape(-1)
    u_ref = jnp.asarray(u_ref).reshape(-1)
    n = u.shape[0]
    return jnp.linalg.norm(u - u_ref) / (u0 * jnp.sqrt(n))
