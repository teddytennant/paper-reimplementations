"""Tiny CPU smoke: rank allocation, decompose/recompose, 20-step linear regression."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from moara.decompose import decompose, recompose
from moara.optim import moara_update
from moara.rank import allocate_ranks


def main() -> None:
    types = ["q", "k", "v", "o", "gate", "up", "down"]
    ranks = allocate_ranks(types, r_base=256)
    print("ranks", ranks, "sum", sum(ranks.values()))

    key = random.PRNGKey(0)
    g = random.normal(key, (32, 48))
    m, v = decompose(g, 16)
    g_hat = recompose(m, v, 16)
    rel = float(jnp.linalg.norm(g - g_hat) / jnp.linalg.norm(g))
    print("recompose relative error", rel)

    k_w, k_x, k_n = random.split(key, 3)
    w_true = random.normal(k_w, (8, 12))
    x = random.normal(k_x, (64, 12))
    y = x @ w_true.T + 0.01 * random.normal(k_n, (64, 8))
    w = jnp.zeros_like(w_true)

    def loss(w_):
        return jnp.mean((x @ w_.T - y) ** 2)

    loss_grad = jax.grad(loss)
    print("loss start", float(loss(w)))
    state = None
    for _ in range(20):
        g_w = loss_grad(w)
        w, state = moara_update(w, g_w, state, block_size=4, rank=4, lr=0.2, scale=1.0, t_svd=5)
    print("loss end", float(loss(w)))


if __name__ == "__main__":
    main()
