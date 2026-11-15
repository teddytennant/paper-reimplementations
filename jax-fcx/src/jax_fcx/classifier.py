"""Frozen two-layer MLP binary classifier (paper §4.1)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


Params = dict[str, Any]


def _init_linear(key: jax.Array, d_in: int, d_out: int, scale: float | None = None) -> dict[str, jax.Array]:
    if scale is None:
        scale = (2.0 / d_in) ** 0.5
    return {"W": jax.random.normal(key, (d_in, d_out)) * scale, "b": jnp.zeros((d_out,))}


def init_classifier(key: jax.Array, d_in: int, hidden: int = 32, n_classes: int = 2) -> Params:
    k1, k2 = jax.random.split(key)
    return {"l1": _init_linear(k1, d_in, hidden), "l2": _init_linear(k2, hidden, n_classes, scale=0.1)}


def forward(params: Params, x: jax.Array) -> jax.Array:
    """Return class logits of shape (..., n_classes)."""
    h = jax.nn.relu(x @ params["l1"]["W"] + params["l1"]["b"])
    return h @ params["l2"]["W"] + params["l2"]["b"]


def predict(params: Params, x: jax.Array) -> jax.Array:
    return jnp.argmax(forward(params, x), axis=-1)


def softmax_cross_entropy(logits: jax.Array, y: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(y.astype(jnp.int32), logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def train_classifier(
    key: jax.Array,
    x: jax.Array,
    y: jax.Array,
    hidden: int = 32,
    steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-2,
    n_classes: int = 2,
) -> tuple[Params, list[float]]:
    """Train the two-layer MLP; caller freezes the returned params."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    n, d = x.shape
    key, k_init = jax.random.split(key)
    params = init_classifier(k_init, d, hidden=hidden, n_classes=n_classes)
    history: list[float] = []

    def loss_fn(p, bx, by):
        return softmax_cross_entropy(forward(p, bx), by)

    @jax.jit
    def step(p, bx, by):
        loss, grads = jax.value_and_grad(loss_fn)(p, bx, by)
        p = jax.tree.map(lambda w, g: w - lr * g, p, grads)
        return p, loss

    bs = min(batch_size, n)
    for t in range(steps):
        key, k_perm = jax.random.split(key)
        idx = jax.random.permutation(k_perm, n)[:bs]
        params, loss = step(params, x[idx], y[idx])
        if t % 10 == 0 or t == steps - 1:
            history.append(float(loss))
    return params, history


def accuracy(params: Params, x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.mean(predict(params, x) == y.astype(jnp.int32))
