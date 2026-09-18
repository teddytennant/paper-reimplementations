"""Adam training of TQSC on synthetic measurement → density-matrix pairs."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jax_tqsc.model import TQSCConfig, forward, init_tqsc
from jax_tqsc.physics import fidelity, frobenius_mse, maximally_mixed

Params = dict[str, Any]


def _adam_init(params: Params) -> dict[str, Any]:
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree.map(jnp.zeros_like, params), "t": jnp.array(0, dtype=jnp.int32)}


def _adam_update(
    params: Params,
    grads: Params,
    state: dict[str, Any],
    lr: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Params, dict[str, Any]]:
    t = state["t"] + 1
    m = jax.tree.map(lambda mm, g: b1 * mm + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda vv, g: b2 * vv + (1.0 - b2) * (g * g), state["v"], grads)
    mhat = jax.tree.map(lambda mm: mm / (1.0 - b1**t), m)
    vhat = jax.tree.map(lambda vv: vv / (1.0 - b2**t), v)
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat)
    return params, {"m": m, "v": v, "t": t}


def loss_fn(params: Params, measurements: jax.Array, rho: jax.Array) -> jax.Array:
    pred = forward(params, measurements)
    return frobenius_mse(pred, rho)


def evaluate(params: Params, measurements: jax.Array, rho: jax.Array) -> dict[str, float]:
    pred = forward(params, measurements)
    fid = fidelity(pred, rho)
    mixed = maximally_mixed(rho.shape[0])
    return {
        "mse": float(frobenius_mse(pred, rho)),
        "fidelity": float(jnp.mean(fid)),
        "mixed_fidelity": float(jnp.mean(fidelity(mixed, rho))),
    }


def train_tqsc(
    key: jax.Array,
    measurements: jax.Array,
    rho: jax.Array,
    cfg: TQSCConfig | None = None,
    steps: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    params: Params | None = None,
) -> tuple[Params, dict[str, Any]]:
    cfg = cfg or TQSCConfig()
    if params is None:
        key, k_init = jax.random.split(key)
        params = init_tqsc(k_init, cfg)
    opt = _adam_init(params)
    m = jnp.asarray(measurements)
    r = jnp.asarray(rho)
    n = int(m.shape[0])
    batch_size = min(batch_size, n)

    def step(params, opt, batch_m, batch_r):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_m, batch_r)
        params, opt = _adam_update(params, grads, opt, lr=lr)
        return params, opt, loss

    step = jax.jit(step)
    losses: list[float] = []
    for _ in range(steps):
        key, k_b = jax.random.split(key)
        idx = jax.random.choice(k_b, n, (batch_size,), replace=False)
        params, opt, loss = step(params, opt, m[idx], r[idx])
        losses.append(float(loss))
    return params, {"loss": losses}
