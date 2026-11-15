"""Tiny linear layers as JAX pytrees."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Linear(NamedTuple):
    W: jax.Array
    b: jax.Array | None


def init_linear(key: jax.Array, in_dim: int, out_dim: int, *, bias: bool = True) -> Linear:
    W = jax.random.normal(key, (in_dim, out_dim)) * jnp.sqrt(1.0 / in_dim)
    b = jnp.zeros((out_dim,)) if bias else None
    return Linear(W, b)


def apply_linear(params: Linear, x: jax.Array) -> jax.Array:
    y = x @ params.W
    if params.b is not None:
        y = y + params.b
    return y
