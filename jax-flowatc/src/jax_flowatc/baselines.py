"""Constant-velocity baseline: last observed (x, y) plus last (vx, vy) * Δt."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def constant_velocity(
    x_obs: Array,
    times_obs: Array,
    times_fut: Array,
) -> Array:
    """Propagate last observed horizontal position with last observed velocity.

    x_obs: (T_obs, ≥6) with features (x, y, z, vx, vy, vz).
    times_obs / times_fut: timestamps of the observed prefix and future suffix.
    Returns predicted future (x, y) of shape (T_fut, 2).
    """
    p = x_obs[-1, :2]
    v = x_obs[-1, 3:5]
    dt = times_fut - times_obs[-1]
    return p + dt[:, None] * v


def constant_velocity_ade(x_obs: Array, x_fut: Array, times_obs: Array, times_fut: Array) -> Array:
    """ADE of the constant-velocity baseline vs ground-truth future (x, y)."""
    pred = constant_velocity(x_obs, times_obs, times_fut)
    d = pred - x_fut[..., :2]
    return jnp.mean(jnp.linalg.norm(d, axis=-1))
