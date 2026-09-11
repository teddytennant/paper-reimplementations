"""Linear and tiny-MLP score functions s: x → R."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Params = dict[str, jax.Array]


def init_linear(key: jax.Array, dim: int, scale: float = 0.1) -> Params:
    w = scale * jax.random.normal(key, (dim,))
    b = jnp.zeros(())
    return {"w": w, "b": b}


def init_mlp(key: jax.Array, dim: int, hidden: int = 8, scale: float = 0.1) -> Params:
    k1, k2 = jax.random.split(key)
    return {
        "W1": scale * jax.random.normal(k1, (dim, hidden)),
        "b1": jnp.zeros((hidden,)),
        "W2": scale * jax.random.normal(k2, (hidden,)),
        "b2": jnp.zeros(()),
    }


def apply_linear(params: Params, x: jax.Array) -> jax.Array:
    return x @ params["w"] + params["b"]


def apply_mlp(params: Params, x: jax.Array) -> jax.Array:
    h = jax.nn.relu(x @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def apply_score(params: Params, x: jax.Array) -> jax.Array:
    if "W1" in params:
        return apply_mlp(params, x)
    return apply_linear(params, x)


class LinearScore:
    """Thin wrapper around a linear score s(x) = w·x + b."""

    def __init__(self, params: Params):
        self.params = params

    @classmethod
    def init(cls, key: jax.Array, dim: int, scale: float = 0.1) -> "LinearScore":
        return cls(init_linear(key, dim, scale=scale))

    def __call__(self, x: jax.Array) -> jax.Array:
        return apply_linear(self.params, x)


class TinyMLP:
    """One-hidden-layer ReLU MLP score."""

    def __init__(self, params: Params):
        self.params = params

    @classmethod
    def init(
        cls, key: jax.Array, dim: int, hidden: int = 8, scale: float = 0.1
    ) -> "TinyMLP":
        return cls(init_mlp(key, dim, hidden=hidden, scale=scale))

    def __call__(self, x: jax.Array) -> jax.Array:
        return apply_mlp(self.params, x)


def sgd_step(params: Params, grads: Params, lr: float) -> Params:
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
