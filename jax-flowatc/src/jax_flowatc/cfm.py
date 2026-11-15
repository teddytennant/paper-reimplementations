"""Conditional Flow Matching interpolant, LogitNormal t, CFM loss, Euler ODE."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_flowatc.config import FlowATCConfig
from jax_flowatc.model import FlowATCParams, velocity_field

Array = jax.Array


def interpolant(x0: Array, x_fut: Array, t: Array) -> Array:
    """x_t = (1 - t) x_0 + t X_fut  (Eq. 12)."""
    t = jnp.asarray(t, dtype=x0.dtype)
    while t.ndim < x0.ndim:
        t = t[..., None]
    return (1.0 - t) * x0 + t * x_fut


def target_velocity(x0: Array, x_fut: Array) -> Array:
    """Constant target velocity X_fut - x_0 along the straight path (Eq. 12)."""
    return x_fut - x0


def sample_logit_normal(
    key: Array,
    shape: Tuple[int, ...] = (),
    mean: float = 1.0,
    std: float = 1.0,
) -> Array:
    """t ~ LogitNormal(1.0, 1.0): sigmoid of N(mean, std^2) in logit space."""
    z = mean + std * jax.random.normal(key, shape)
    return jax.nn.sigmoid(z)


def cfm_loss_one(
    params: FlowATCParams,
    cfg: FlowATCConfig,
    x_obs: Array,
    x_fut: Array,
    t: Array,
    x0: Array,
    dt: Array,
    t_rel: Array,
    dropout_key: Optional[Array] = None,
) -> Array:
    """||u_θ(x_t, t | X_obs) - (X_fut - x_0)||_2^2 averaged over tokens and dims."""
    x_t = interpolant(x0, x_fut, t)
    target = target_velocity(x0, x_fut)
    pred = velocity_field(params, cfg, x_obs, x_t, t, dt, t_rel, dropout_key)
    err = pred - target
    return jnp.mean(err * err)


def cfm_loss_batch(
    params: FlowATCParams,
    cfg: FlowATCConfig,
    x_obs: Array,
    x_fut: Array,
    dt: Array,
    t_rel: Array,
    key: Array,
) -> Array:
    """Sample t ~ LogitNormal and x0 ~ N(0, I) per batch element, mean CFM loss."""
    b = x_obs.shape[0]
    k_t, k_x, k_drop = jax.random.split(key, 3)
    t = sample_logit_normal(k_t, (b,))
    x0 = jax.random.normal(k_x, x_fut.shape, dtype=x_fut.dtype)
    drop_keys = jax.random.split(k_drop, b) if cfg.dropout > 0.0 else jnp.zeros((b, 2), dtype=jnp.uint32)

    def one(xo, xf, ti, x0i, dti, tri, dk):
        dk_use = dk if cfg.dropout > 0.0 else None
        return cfm_loss_one(params, cfg, xo, xf, ti, x0i, dti, tri, dk_use)

    losses = jax.vmap(one)(x_obs, x_fut, t, x0, dt, t_rel, drop_keys)
    return jnp.mean(losses)


def euler_integrate(u_fn: Callable[[Array, Array], Array], x0: Array, n_steps: int) -> Array:
    """Euler from t=0 to t=1 in n_steps: x <- x + (1/N) u(x, t)."""
    dt = 1.0 / n_steps

    def body(i, x):
        t = jnp.asarray(i, dtype=x.dtype) * jnp.asarray(dt, dtype=x.dtype)
        return x + dt * u_fn(x, t)

    return jax.lax.fori_loop(0, n_steps, body, x0)


def euler_path(u_fn: Callable[[Array, Array], Array], x0: Array, n_steps: int) -> Array:
    """All Euler states after each step, shape (n_steps, ...)."""
    dt = 1.0 / n_steps

    def body(x, i):
        t = jnp.asarray(i, dtype=x.dtype) * jnp.asarray(dt, dtype=x.dtype)
        x_next = x + dt * u_fn(x, t)
        return x_next, x_next

    _, xs = jax.lax.scan(body, x0, jnp.arange(n_steps))
    return xs


def sample_completions(
    params: FlowATCParams,
    cfg: FlowATCConfig,
    x_obs: Array,
    dt: Array,
    t_rel: Array,
    key: Array,
    n_steps: int = 20,
    k: int = 1,
) -> Array:
    """K independent Euler completions. Returns (K, T_fut, state_dim)."""
    keys = jax.random.split(key, k)

    def one(rng):
        x0 = jax.random.normal(rng, (cfg.t_fut, cfg.state_dim), dtype=x_obs.dtype)

        def u_fn(x, t):
            return velocity_field(params, cfg, x_obs, x, t, dt, t_rel, None)

        return euler_integrate(u_fn, x0, n_steps)

    return jax.vmap(one)(keys)
