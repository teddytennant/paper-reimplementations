"""Tiny CPU smoke train on synthetic 16-joint sequences."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from svgcn.data import generate_synthetic
from svgcn.model import apply_svgcn, cross_entropy, init_svgcn


def _adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def _adam_update(params, grads, state, lr=8e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * (g * g), state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat
    )
    return params, {"m": m, "v": v, "t": t}


def main() -> None:
    key = jax.random.PRNGKey(0)
    k_data, k_model = jax.random.split(key)
    coords, lengths, labels = generate_synthetic(
        k_data, batch_size=64, t_max=32, min_len=12
    )
    params = init_svgcn(k_model)
    opt_state = _adam_init(params)

    def loss_fn(p, x, ln, y):
        logits = apply_svgcn(p, x, ln)
        return cross_entropy(logits, y), logits

    @jax.jit
    def step(p, st, x, ln, y):
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, x, ln, y)
        p, st = _adam_update(p, grads, st)
        acc = jnp.mean((jnp.argmax(logits, axis=-1) == y).astype(jnp.float32))
        return p, st, loss, acc

    n_steps = 50
    for i in range(n_steps):
        params, opt_state, loss, acc = step(params, opt_state, coords, lengths, labels)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == n_steps:
            print(f"step {i+1:02d}/{n_steps}  loss={float(loss):.4f}  acc={float(acc):.3f}")
    print(f"final train acc={float(acc):.3f}")


if __name__ == "__main__":
    main()
