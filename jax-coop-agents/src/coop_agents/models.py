"""Tiny explicit-pytree MLP (no Flax/Haiku)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp


Params = tuple[dict[str, jax.Array], ...]
InitFn = Callable[[jax.Array], Params]
ApplyFn = Callable[[Params, jax.Array], jax.Array]


def mlp(sizes: Sequence[int]) -> tuple[InitFn, ApplyFn]:
    """Dense ReLU MLP with linear logits. `sizes` is [in, hidden..., out]."""
    sizes = tuple(int(s) for s in sizes)

    def init(key: jax.Array) -> Params:
        layers = []
        for in_f, out_f in zip(sizes[:-1], sizes[1:]):
            key, k_w = jax.random.split(key)
            w = jax.random.normal(k_w, (in_f, out_f)) * jnp.sqrt(2.0 / in_f)
            b = jnp.zeros((out_f,))
            layers.append({"W": w, "b": b})
        return tuple(layers)

    return init, apply_mlp


def apply_mlp(params: Params, x: jax.Array) -> jax.Array:
    """Forward pass: ReLU hidden layers, linear output (logits)."""
    h = jnp.asarray(x)
    n_layers = len(params)
    for i, layer in enumerate(params):
        h = h @ layer["W"] + layer["b"]
        if i < n_layers - 1:
            h = jax.nn.relu(h)
    return h


def softmax(logits: jax.Array) -> jax.Array:
    return jax.nn.softmax(logits, axis=-1)


def count_params(params: Params) -> int:
    leaves = jax.tree.leaves(params)
    return int(sum(int(x.size) for x in leaves))
