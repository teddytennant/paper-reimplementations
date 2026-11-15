"""Tiny MLP smoke train with Soft Musec on synthetic data."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from musec.optimizer import init, step


def mlp_init(key, in_dim=8, hidden=16, out_dim=4):
    k1, k2, k3 = jax.random.split(key, 3)
    scale = lambda k, shape: jax.random.normal(k, shape) * (2.0 / shape[0]) ** 0.5
    return {
        "W1": scale(k1, (hidden, in_dim)),
        "b1": jnp.zeros(hidden),
        "W2": scale(k2, (out_dim, hidden)),
        "b2": jnp.zeros(out_dim),
    }


def mlp_apply(params, x):
    h = jnp.tanh(x @ params["W1"].T + params["b1"])
    return h @ params["W2"].T + params["b2"]


def main():
    key = jax.random.PRNGKey(0)
    n, in_dim, out_dim = 64, 8, 4
    k_x, k_w, k_p = jax.random.split(key, 3)
    x = jax.random.normal(k_x, (n, in_dim))
    true_w = jax.random.normal(k_w, (in_dim, out_dim))
    y = x @ true_w

    params = mlp_init(k_p, in_dim=in_dim, hidden=16, out_dim=out_dim)
    state = init(params)

    def loss_fn(p):
        pred = mlp_apply(p, x)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def train_step(params, state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, state = step(
            params,
            grads,
            state,
            eta=0.05,
            beta=0.9,
            D=1.0,
            mode="soft",
            ns_steps=5,
        )
        return params, state, loss

    t0 = time.time()
    last = None
    for i in range(20):
        params, state, last = train_step(params, state)
        last.block_until_ready()
        if i == 0 or i == 19:
            print(f"step {i:02d} loss={float(last):.6f}")
    print(f"elapsed_s={time.time() - t0:.3f}")
    if not jnp.isfinite(last):
        raise SystemExit("non-finite loss")


if __name__ == "__main__":
    main()
