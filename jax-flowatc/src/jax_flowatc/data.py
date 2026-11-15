"""Z-score per feature and synthetic irregular trajectories (no ADS-B)."""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


class ZScoreStats(NamedTuple):
    mean: Array
    std: Array


def zscore_stats(x: Array, eps: float = 1e-6) -> ZScoreStats:
    """Independent per-feature mean/std over all tokens in x (..., D)."""
    flat = x.reshape(-1, x.shape[-1])
    mean = jnp.mean(flat, axis=0)
    std = jnp.std(flat, axis=0)
    return ZScoreStats(mean=mean, std=jnp.maximum(std, eps))


def apply_zscore(x: Array, stats: ZScoreStats) -> Array:
    return (x - stats.mean) / stats.std


def invert_zscore(x: Array, stats: ZScoreStats) -> Array:
    return x * stats.std + stats.mean


def deltas_and_elapsed(times: Array) -> Tuple[Array, Array]:
    """dt[0]=0, dt[i]=t_i-t_{i-1}; t_rel[i]=t_i-t_0."""
    t_rel = times - times[..., :1]
    zeros = jnp.zeros(times.shape[:-1] + (1,), dtype=times.dtype)
    dt = jnp.concatenate([zeros, jnp.diff(times, axis=-1)], axis=-1)
    return dt, t_rel


def synthetic_straight(
    key: Array,
    batch: int,
    t_obs: int,
    t_fut: int,
    state_dim: int = 6,
) -> Tuple[Array, Array]:
    """Straight-line flights with irregular Δt. Returns states (B, T, D), times (B, T)."""
    t_len = t_obs + t_fut
    k_p, k_v, k_dt = jax.random.split(key, 3)
    p0 = jax.random.normal(k_p, (batch, 3)) * 50.0
    vel = jax.random.normal(k_v, (batch, 3)) * 20.0 + jnp.array([80.0, 10.0, 1.0])
    dt = 0.5 + 3.5 * jax.random.uniform(k_dt, (batch, t_len))
    dt = dt.at[:, 0].set(0.0)
    times = jnp.cumsum(dt, axis=1)
    pos = p0[:, None, :] + times[:, :, None] * vel[:, None, :]
    vel_tok = jnp.broadcast_to(vel[:, None, :], pos.shape)
    xyzv = jnp.concatenate([pos, vel_tok], axis=-1)
    if state_dim != 6:
        xyzv = xyzv[..., :state_dim]
    return xyzv, times


def synthetic_with_turn(
    key: Array,
    batch: int,
    t_obs: int,
    t_fut: int,
    state_dim: int = 6,
    turn_rad: float = 0.6,
) -> Tuple[Array, Array]:
    """Constant velocity through the observed prefix, then a heading change in the future."""
    t_len = t_obs + t_fut
    k_p, k_v, k_dt, k_turn = jax.random.split(key, 4)
    p0 = jax.random.normal(k_p, (batch, 3)) * 30.0
    speed = 70.0 + 20.0 * jax.random.uniform(k_v, (batch,))
    heading = jax.random.uniform(k_turn, (batch,), minval=-0.4, maxval=0.4)
    vx = speed * jnp.cos(heading)
    vy = speed * jnp.sin(heading)
    vz = jnp.zeros((batch,))
    vel0 = jnp.stack([vx, vy, vz], axis=-1)
    dt = 0.5 + 3.5 * jax.random.uniform(k_dt, (batch, t_len))
    dt = dt.at[:, 0].set(0.0)
    times = jnp.cumsum(dt, axis=1)

    def one(p, v, ts):
        def step(carry, t_i):
            pos, vel, t_prev = carry
            dti = t_i - t_prev
            pos = pos + dti * vel
            return (pos, vel, t_i), (pos, vel)

        # turn applied from the first future token onward
        c0 = (p, v, ts[0])
        _, (pos_obs, vel_obs) = jax.lax.scan(step, c0, ts[:t_obs])
        # reset scan from last observed
        last_p = pos_obs[-1]
        last_v = vel_obs[-1]
        c, s = jnp.cos(turn_rad), jnp.sin(turn_rad)
        rot = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        v_turn = rot @ last_v
        c1 = (last_p, v_turn, ts[t_obs - 1])
        _, (pos_fut, vel_fut) = jax.lax.scan(step, c1, ts[t_obs:])
        pos = jnp.concatenate([pos_obs, pos_fut], axis=0)
        vel = jnp.concatenate([vel_obs, vel_fut], axis=0)
        return jnp.concatenate([pos, vel], axis=-1)

    states = jax.vmap(one)(p0, vel0, times)
    if state_dim != 6:
        states = states[..., :state_dim]
    return states, times
