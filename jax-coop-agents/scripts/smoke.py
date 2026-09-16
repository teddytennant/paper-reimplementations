"""CPU smoke: 4-class 2D blobs, 3 small agents vs 1 large, 50 SGD steps."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from coop_agents.data import gaussian_blobs
from coop_agents.losses import isolated_loss
from coop_agents.models import apply_mlp, count_params, mlp, softmax
from coop_agents.train import init_opt_state, step


def accuracy(params, x, y):
    pred = jnp.argmax(apply_mlp(params, x), axis=-1)
    true = jnp.argmax(y, axis=-1)
    return float(jnp.mean((pred == true).astype(jnp.float32)))


def mean_isol(agents, x, y):
    vals = [float(jnp.mean(isolated_loss(softmax(apply_mlp(p, x)), y))) for p in agents]
    return float(sum(vals) / len(vals))


def mean_acc(agents, x, y):
    return float(sum(accuracy(p, x, y) for p in agents) / len(agents))


def train_loop(agents, x, y, strategy, steps=50, lr=0.2, seed=0):
    opt = init_opt_state(agents, lr=lr, kind="sgd", key=jax.random.PRNGKey(seed))
    start_loss = mean_isol(agents, x, y)
    start_acc = mean_acc(agents, x, y)
    for _ in range(steps):
        agents, opt, _ = step(agents, (x, y), strategy, 1.0, opt)
    end_loss = mean_isol(agents, x, y)
    end_acc = mean_acc(agents, x, y)
    return start_loss, end_loss, start_acc, end_acc


def main():
    key = jax.random.PRNGKey(0)
    x, y = gaussian_blobs(key, n=64, n_classes=4, dim=2, std=0.2, scale=2.0)

    init_small, _ = mlp([2, 8, 4])
    init_large, _ = mlp([2, 32, 4])
    sk = jax.random.split(jax.random.PRNGKey(1), 4)
    small_template = [init_small(k) for k in sk[:3]]
    large = [init_large(sk[3])]

    n_small = 3 * count_params(small_template[0])
    n_large = count_params(large[0])
    print(f"param counts: 3 small MLPs (hidden 8) = {n_small}; 1 large MLP (hidden 32) = {n_large}")
    assert n_small < n_large

    strategies = ("isolated", "voter", "majority", "weighted_average")
    for name in strategies:
        agents = [init_small(k) for k in sk[:3]]
        s_loss, e_loss, s_acc, e_acc = train_loop(agents, x, y, name, steps=50, lr=0.05, seed=10)
        print(
            f"{name:18s}  loss {s_loss:.4f} -> {e_loss:.4f}   "
            f"acc {s_acc:.3f} -> {e_acc:.3f}"
        )

    s_loss, e_loss, s_acc, e_acc = train_loop(large, x, y, "isolated", steps=50, lr=0.05, seed=11)
    print(
        f"{'large-isolated':18s}  loss {s_loss:.4f} -> {e_loss:.4f}   "
        f"acc {s_acc:.3f} -> {e_acc:.3f}"
    )


if __name__ == "__main__":
    main()
