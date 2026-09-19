"""Tiny AdamW and a jittable CFM train step. No optax."""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from jax_flowatc.cfm import cfm_loss_batch
from jax_flowatc.config import FlowATCConfig
from jax_flowatc.model import FlowATCParams

Array = jax.Array


class AdamWState(NamedTuple):
    m: FlowATCParams
    v: FlowATCParams
    step: Array


def adamw_init(params: FlowATCParams) -> AdamWState:
    zeros = jax.tree.map(jnp.zeros_like, params)
    return AdamWState(m=zeros, v=zeros, step=jnp.array(0, dtype=jnp.int32))


def _is_weight(x: Array) -> bool:
    return x.ndim >= 2


def adamw_update(
    params: FlowATCParams,
    grads: FlowATCParams,
    state: AdamWState,
    lr: float = 1e-4,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    wd: float = 1e-2,
) -> Tuple[FlowATCParams, AdamWState]:
    step = state.step + 1
    m = jax.tree.map(lambda mi, g: b1 * mi + (1.0 - b1) * g, state.m, grads)
    v = jax.tree.map(lambda vi, g: b2 * vi + (1.0 - b2) * (g * g), state.v, grads)
    mhat = jax.tree.map(lambda mi: mi / (1.0 - b1 ** step), m)
    vhat = jax.tree.map(lambda vi: vi / (1.0 - b2 ** step), v)

    def apply(p, mh, vh, g):
        update = mh / (jnp.sqrt(vh) + eps)
        decay = wd * p if _is_weight(p) else 0.0
        return p - lr * (update + decay)

    new_params = jax.tree.map(apply, params, mhat, vhat, grads)
    return new_params, AdamWState(m=m, v=v, step=step)


def train_step(
    params: FlowATCParams,
    opt_state: AdamWState,
    x_obs: Array,
    x_fut: Array,
    dt: Array,
    t_rel: Array,
    key: Array,
    cfg: FlowATCConfig,
    lr: float = 1e-4,
    wd: float = 1e-2,
) -> Tuple[FlowATCParams, AdamWState, Array, FlowATCParams]:
    """One AdamW step. Returns params, opt_state, loss, grads."""

    def loss_fn(p):
        return cfm_loss_batch(p, cfg, x_obs, x_fut, dt, t_rel, key)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params, opt_state = adamw_update(params, grads, opt_state, lr=lr, wd=wd)
    return params, opt_state, loss, grads


def make_train_step(cfg: FlowATCConfig, lr: float = 1e-4, wd: float = 1e-2):
    """Jitted train_step with config/lr/wd closed over (static)."""

    def step(params, opt_state, x_obs, x_fut, dt, t_rel, key):
        return train_step(params, opt_state, x_obs, x_fut, dt, t_rel, key, cfg, lr=lr, wd=wd)

    return jax.jit(step)
