"""A small MLP in raw JAX.

Both the surrogate ensemble of Eq. (23) and the noise-prediction network of the
trajectory-modeling stage are built from this. Parameters are a list of
(weight, bias) pairs, so an ensemble is the same pytree with a leading member
axis and ``jax.vmap`` over it.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

__all__ = ["init_mlp", "apply_mlp", "leaky_relu"]


def leaky_relu(x: jnp.ndarray, slope: float = 0.01) -> jnp.ndarray:
    return jnp.where(x >= 0, x, slope * x)


def init_mlp(key: jax.Array, sizes: Sequence[int]) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """He-style init for a fully connected stack with layer widths ``sizes``."""
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, fan_in, fan_out in zip(keys, sizes[:-1], sizes[1:]):
        scale = jnp.sqrt(2.0 / fan_in)
        params.append((jax.random.normal(k, (fan_in, fan_out)) * scale, jnp.zeros((fan_out,))))
    return params


def apply_mlp(params, x: jnp.ndarray, slope: float = 0.01) -> jnp.ndarray:
    """Forward pass with LeakyReLU between layers and a linear output."""
    for W, b in params[:-1]:
        x = leaky_relu(x @ W + b, slope)
    W, b = params[-1]
    return x @ W + b
