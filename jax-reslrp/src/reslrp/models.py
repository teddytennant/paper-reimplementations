"""Tiny residual MLP: x → x + W2 gelu(W1 x), two blocks, no pretrained weights."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random


def init_residual_mlp(
    key,
    d_in: int = 16,
    d_hidden: int = 32,
    n_classes: int = 4,
    n_blocks: int = 2,
    scale: float = 0.3,
):
    """Random parameters for a bias-free 2-block residual MLP."""
    keys = random.split(key, 2 * n_blocks + 1)
    blocks = []
    for i in range(n_blocks):
        k1, k2 = keys[2 * i], keys[2 * i + 1]
        blocks.append(
            {
                "W1": scale * random.normal(k1, (d_in, d_hidden)),
                "W2": scale * random.normal(k2, (d_hidden, d_in)),
            }
        )
    w_head = scale * random.normal(keys[-1], (d_in, n_classes))
    return {"blocks": blocks, "W_head": w_head}


def residual_mlp(params, x, cache: bool = False):
    """Forward. If ``cache``, also return per-block z_in, z_up, z_out, hid."""
    caches = []
    h = x
    for block in params["blocks"]:
        z_in = h
        hid = jax.nn.gelu(z_in @ block["W1"])
        z_up = hid @ block["W2"]
        z_out = z_in + z_up
        if cache:
            caches.append({"z_in": z_in, "z_up": z_up, "z_out": z_out, "hid": hid})
        h = z_out
    logits = h @ params["W_head"]
    if cache:
        return logits, caches
    return logits
