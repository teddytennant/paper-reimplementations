"""Tiny stack of identifiable affine layers.

Each layer is its own parameter group: h -> h @ W + b. Layer ids are
1-indexed. The forward pass always runs every layer. Cached prefix
activations are not used. Truncated backward stops the gradient on the
input of G_k, at l_start.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def init_affine_stack(depth: int, dim: int, key, scale: float = 0.05):
    """Near-identity affine stack. Bias entry 0 stores the layer id."""

    if depth < 1:
        raise ValueError("depth must be at least 1")
    if dim < 1:
        raise ValueError("dim must be at least 1")
    keys = jax.random.split(key, depth)
    params = {}
    for offset, layer_key in enumerate(keys):
        layer_id = offset + 1
        k_w, k_b = jax.random.split(layer_key)
        eye = jnp.eye(dim, dtype=jnp.float32)
        noise = scale * jax.random.normal(k_w, (dim, dim), dtype=jnp.float32)
        weight = eye + noise
        bias = scale * jax.random.normal(k_b, (dim,), dtype=jnp.float32)
        bias = bias.at[0].set(jnp.float32(layer_id))
        params[layer_id] = {"w": weight, "b": bias}
    return params


def forward(params, x, layer_ids, l_start):
    """Full forward pass. Stop-gradient on the input of the layer l_start."""

    hidden = x
    for layer_id in layer_ids:
        if layer_id == l_start:
            hidden = jax.lax.stop_gradient(hidden)
        layer = params[layer_id]
        hidden = hidden @ layer["w"] + layer["b"]
    return hidden


def loss_and_grads(params, x, y, layer_ids, l_start):
    """Mean squared error and gradients under truncated backward."""

    def loss_fn(current):
        pred = forward(current, x, layer_ids, l_start)
        return jnp.mean((pred - y) ** 2)

    return jax.value_and_grad(loss_fn)(params)


def copy_params(params):
    """Deep copy of a layer-id parameter dict."""

    return {
        layer_id: {name: jnp.array(np.array(value)) for name, value in layer.items()}
        for layer_id, layer in params.items()
    }
