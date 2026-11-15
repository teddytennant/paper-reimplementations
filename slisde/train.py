"""Synthetic functional-calibration smoke for SLiSDE (CPU)."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from slisde._host import preload

preload()

import jax
import jax.numpy as jnp

from slisde.loss import calibration_loss, path_mean, path_mean_square, terminal
from slisde.model import init_model, sample_z0, simulate
from slisde.sde import sample_increments


def _adam_init(params, lr):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32), "lr": lr}


def _adam_update(params, grads, state, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    lr = state["lr"]
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1.0 - b2) * g * g, state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda mm: mm / (1.0 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda vv: vv / (1.0 - b2 ** t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat
    )
    return params, {"m": m, "v": v, "t": t, "lr": lr}


def run_smoke(
    *,
    steps=40,
    batch=64,
    n_layers=2,
    d=4,
    n_steps=8,
    T=1.0,
    seed=0,
    lr=0.08,
    print_loss=True,
):
    dt = T / n_steps
    key = jax.random.PRNGKey(seed)
    key, k_model = jax.random.split(key)
    params = init_model(k_model, n_layers=n_layers, d=d, m=1, p=1, noise="additive")
    targets = (jnp.array(0.5), jnp.array(0.25), jnp.array(0.4))

    def loss_fn(p, z0, dW):
        Y, _, _ = simulate(p, z0, dW, dt, parallel=True)
        estimates = (jnp.mean(terminal(Y)), jnp.mean(path_mean(Y)), jnp.mean(path_mean_square(Y)))
        return calibration_loss(estimates, targets)

    opt_state = _adam_init(params, lr)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    def step(params, opt_state, key):
        key, k0, kW = jax.random.split(key, 3)
        z0 = sample_z0(k0, batch, d)
        dW = sample_increments(kW, batch, n_steps, 1, dt)
        loss, grads = loss_and_grad(params, z0, dW)
        params, opt_state = _adam_update(params, grads, opt_state)
        return params, opt_state, key, loss

    params, opt_state, key, start = step(params, opt_state, key)
    start = float(start)
    if print_loss:
        print(f"start_loss={start:.6f}")

    last = start
    for _ in range(steps - 1):
        params, opt_state, key, last = step(params, opt_state, key)
    end = float(last)
    if print_loss:
        print(f"end_loss={end:.6f}")
    return start, end


def main():
    run_smoke()


if __name__ == "__main__":
    main()
