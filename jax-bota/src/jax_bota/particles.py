"""Discrete-time particle branched OT (Sec. 5.1 of arXiv:2609.15072).

Euler discretization (documented default; paper writes x_{k+1}^i = x_k^i +
V_{k,i} Delta t with Delta t = 1/T, but does not publish T, sigma, lambda,
or the proxy-center schedule):

    x_{k+1} = x_k + v_k * dt,   dt = 1/T,   k = 0, ..., T-1
    x_0 given (source samples)

Discrete objective (Sec. 5.1):

    L_discrete(V) = sum_{k=0}^{T-1} C_soft(X_k, V_k) dt
                    + lambda * (1/N) sum_i ||x_T^i - x_1^i||^2

Particle masses are not updated by Euler: they stay the input masses
(non-negative, conserved sum). Soft mass is only a readout onto proxy atoms.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_bota.cost import (
    SpeedMode,
    bounding_grid_centers,
    integrated_soft_cost,
    soft_atomic_cost,
)


class DiscreteSolve(NamedTuple):
    velocities: jax.Array
    positions: jax.Array
    masses: jax.Array
    loss: jax.Array
    transport_cost: jax.Array
    terminal_cost: jax.Array


def normalize_masses(masses: jax.Array, total: float = 1.0) -> jax.Array:
    """Clip to non-negative and rescale so sum(m) = total."""
    clipped = jnp.clip(masses, 0.0)
    s = jnp.sum(clipped)
    n = masses.shape[0]
    uniform = jnp.full_like(masses, total / n)
    return jnp.where(s > 0, clipped * (total / s), uniform)


def euler_trajectories(
    x0: jax.Array,
    velocities: jax.Array,
    dt: float,
) -> jax.Array:
    """Integrate x_{k+1} = x_k + v_k dt.

    Args:
        x0: (N, D)
        velocities: (T, N, D)
        dt: 1/T

    Returns:
        positions of shape (T+1, N, D), including x0.
    """

    def step(x, v):
        x_next = x + v * dt
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, velocities)
    return jnp.concatenate([x0[None, ...], xs], axis=0)


def terminal_mse(x_t: jax.Array, x_target: jax.Array) -> jax.Array:
    """(1/N) sum_i ||x_T^i - x_1^i||^2 as in Sec. 5.1."""
    err = x_t - x_target
    return jnp.mean(jnp.sum(err * err, axis=-1))


def discrete_loss(
    velocities: jax.Array,
    x0: jax.Array,
    x_target: jax.Array,
    centers: jax.Array,
    alpha: float,
    dt: float,
    sigma: float = 0.5,
    lam: float = 5.0,
    masses: jax.Array | None = None,
    speed_mode: SpeedMode = "speed",
) -> jax.Array:
    """L_discrete(V) from Sec. 5.1."""
    positions = euler_trajectories(x0, velocities, dt)
    transport = integrated_soft_cost(
        positions,
        velocities,
        centers,
        alpha,
        dt,
        sigma=sigma,
        masses=masses,
        speed_mode=speed_mode,
    )
    return transport + lam * terminal_mse(positions[-1], x_target)


def straight_velocities(x0: jax.Array, x_target: jax.Array, n_steps: int) -> jax.Array:
    """Constant velocity that interpolates x0 -> x_target in unit time."""
    v = x_target - x0
    return jnp.repeat(v[None, ...], n_steps, axis=0)


def solve_discrete(
    x0: jax.Array,
    x_target: jax.Array,
    alpha: float = 0.5,
    n_steps: int = 8,
    n_opt: int = 40,
    lr: float = 0.08,
    sigma: float = 0.5,
    lam: float = 5.0,
    n_centers: int = 4,
    masses: jax.Array | None = None,
    centers: jax.Array | None = None,
    speed_mode: SpeedMode = "speed",
) -> DiscreteSolve:
    """SGD on particle velocities for L_discrete.

    Optimizer, learning rate, T, sigma, lambda, and center grid are not
    published in the paper; this uses Euler + plain gradient descent.
    """
    n, _ = x0.shape
    dt = 1.0 / n_steps
    if masses is None:
        masses = jnp.full((n,), 1.0 / n, dtype=x0.dtype)
    else:
        masses = normalize_masses(masses, total=float(jnp.sum(masses)))
    if centers is None:
        centers = bounding_grid_centers(
            jnp.concatenate([x0, x_target], axis=0),
            n_per_axis=n_centers,
        )
    velocities = straight_velocities(x0, x_target, n_steps)

    def loss_fn(v):
        return discrete_loss(
            v,
            x0,
            x_target,
            centers,
            alpha,
            dt,
            sigma=sigma,
            lam=lam,
            masses=masses,
            speed_mode=speed_mode,
        )

    def opt_step(v, _):
        g = jax.grad(loss_fn)(v)
        return v - lr * g, None

    velocities, _ = jax.lax.scan(opt_step, velocities, xs=None, length=n_opt)
    positions = euler_trajectories(x0, velocities, dt)
    loss = loss_fn(velocities)
    transport = integrated_soft_cost(
        positions,
        velocities,
        centers,
        alpha,
        dt,
        sigma=sigma,
        masses=masses,
        speed_mode=speed_mode,
    )
    terminal = lam * terminal_mse(positions[-1], x_target)
    return DiscreteSolve(
        velocities=velocities,
        positions=positions,
        masses=masses,
        loss=loss,
        transport_cost=transport,
        terminal_cost=terminal,
    )


def snapshot_cost(
    positions: jax.Array,
    velocities: jax.Array,
    centers: jax.Array,
    alpha: float,
    sigma: float = 0.5,
    masses: jax.Array | None = None,
    speed_mode: SpeedMode = "speed",
) -> jax.Array:
    """C_soft at a single time index (for tests / debugging)."""
    return soft_atomic_cost(
        positions,
        velocities,
        centers,
        alpha,
        sigma=sigma,
        masses=masses,
        speed_mode=speed_mode,
    )
