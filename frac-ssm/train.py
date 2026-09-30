"""CPU/GPU-agnostic smoke: delayed EMA toy, few Adam steps, JSON losses."""

from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from frac_ssm._host import preload

preload()

import jax
import jax.numpy as jnp

from frac_ssm.layer import apply_layer, init_frac_layer

D = 16
H = 2
M = 4
STEPS = 40
BATCH = 8
T = 16
LR = 1e-2
SEED = 0
RHO = 0.7


def ema_target(x, rho=RHO):
    def step(h, xt):
        h = rho * h + (1.0 - rho) * xt
        return h, h

    _, y = jax.lax.scan(step, jnp.zeros_like(x[:, 0]), jnp.swapaxes(x, 0, 1))
    return jnp.swapaxes(y, 0, 1)


def adam_init(params, lr):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32), "lr": lr}


def clip_grads(grads, max_norm=1.0):
    leaves = jax.tree_util.tree_leaves(grads)
    gsq = sum((jnp.sum(jnp.square(leaf)) for leaf in leaves), jnp.array(0.0))
    scale = jnp.minimum(1.0, max_norm / (jnp.sqrt(gsq) + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def adam_update(params, grads, state, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    lr = state["lr"]
    grads = clip_grads(grads)
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1.0 - b2) * g * g, state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda mm: mm / (1.0 - b1**t), m)
    vhat = jax.tree_util.tree_map(lambda vv: vv / (1.0 - b2**t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat
    )
    return params, {"m": m, "v": v, "t": t, "lr": lr}


def main() -> None:
    key = jax.random.PRNGKey(SEED)
    k_data, k_model = jax.random.split(key)
    x = jax.random.normal(k_data, (BATCH, T, D))
    y = ema_target(x)
    params = init_frac_layer(k_model, d=D, n_heads=H, n_modes=M)
    opt = adam_init(params, LR)

    def loss_fn(p):
        pred, _ = apply_layer(p, x, parallel=True)
        return jnp.mean(jnp.square(pred - y))

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    start = float(loss_fn(params))
    for _ in range(STEPS):
        _, grads = loss_and_grad(params)
        params, opt = adam_update(params, grads, opt)
    end = float(loss_fn(params))
    results = {
        "loss_start": start,
        "loss_end": end,
        "steps": STEPS,
        "d": D,
        "heads": H,
        "modes": M,
        "T": T,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
