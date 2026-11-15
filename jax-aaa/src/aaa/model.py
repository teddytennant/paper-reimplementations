"""Tiny AXON encoder: stacked AAA blocks plus a linear classification head.

Tokens are already on a C x T grid in R^d (patch projection is assumed
done). Additive channel and time embeddings supply positional encoding.
The encoder is a few blocks, not the 22-layer EEG MAE from the paper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aaa.attention import init_linear
from aaa.block import aaa_block, init_block_params


def init_axon_params(
    key,
    *,
    c,
    t,
    d,
    num_blocks=2,
    num_classes=2,
    gate="gated",
):
    k_pos, k_head, *k_blocks = jax.random.split(key, num_blocks + 2)
    kc, kt = jax.random.split(k_pos)
    blocks = tuple(
        init_block_params(kb, d, gate=gate) for kb in k_blocks[:num_blocks]
    )
    return {
        "channel_emb": jax.random.normal(kc, (c, d)) * 0.02,
        "time_emb": jax.random.normal(kt, (t, d)) * 0.02,
        "blocks": blocks,
        "head_w": init_linear(k_head, d, num_classes),
        "head_b": jnp.zeros((num_classes,), dtype=jnp.float32),
    }


def add_position(params, x):
    return x + params["channel_emb"][None, :, None, :] + params["time_emb"][None, None, :, :]


def axon_encode(params, x, num_heads, knn_mask=None, tau=1.0, path="both"):
    """Stack of AAA blocks. x and output: (B, C, T, D)."""
    h = add_position(params, x)
    for block in params["blocks"]:
        h = aaa_block(
            block,
            h,
            num_heads,
            knn_mask=knn_mask,
            tau=tau,
            path=path,
        )
    return h


def axon_classify(params, x, num_heads, knn_mask=None, tau=1.0, path="both"):
    """Mean-pool the grid and apply a linear head. Returns (B, num_classes)."""
    h = axon_encode(
        params, x, num_heads, knn_mask=knn_mask, tau=tau, path=path
    )
    pooled = jnp.mean(h, axis=(1, 2))
    return pooled @ params["head_w"] + params["head_b"]
