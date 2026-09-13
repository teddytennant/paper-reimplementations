"""Tiny MLP 16-8-1 on synthetic y=sin(x)."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from adamx.opt import apply_updates, init, update

BATCH = 32
STEPS = 40
LR = 0.05


def init_params(key):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (16, 8)) * 0.2,
        "b1": jnp.zeros((8,)),
        "w2": jax.random.normal(k2, (8, 1)) * 0.2,
        "b2": jnp.zeros((1,)),
    }


def mlp(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    return h @ params["w2"] + params["b2"]


def loss_fn(params, x, y):
    pred = mlp(params, x)
    return jnp.mean((pred - y) ** 2)


def main() -> int:
    key = jax.random.PRNGKey(0)
    key, k_x, k_p = jax.random.split(key, 3)
    x = jax.random.normal(k_x, (BATCH, 16))
    y = jnp.sin(x[:, :1])
    params = init_params(k_p)
    state = init(params, lr=LR)
    loss_before = float(loss_fn(params, x, y))
    for _ in range(STEPS):
        grads = jax.grad(loss_fn)(params, x, y)
        updates, state = update(grads, state, params)
        params = apply_updates(params, updates)
    loss_after = float(loss_fn(params, x, y))
    print(f"loss_before={loss_before:.6f}")
    print(f"loss_after={loss_after:.6f}")
    if not (loss_after < loss_before):
        print("loss did not drop", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
