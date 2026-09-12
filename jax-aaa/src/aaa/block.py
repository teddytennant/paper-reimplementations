"""One AXON / AAA block: parallel paths, gate, residual, LayerNorm, FFN.

Paths run in parallel (not sequential axial attention). After the convex
mixture: residual + LayerNorm + FFN, as a Transformer block. Multi-head
splits d into H heads independently on each path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aaa.attention import (
    init_attention_params,
    init_linear,
    spatial_attention,
    temporal_attention,
)
from aaa.gate import fixed_mix, gated_mix, init_fixed_gate, init_token_gate


def init_layer_norm(d):
    return {
        "scale": jnp.ones((d,), dtype=jnp.float32),
        "bias": jnp.zeros((d,), dtype=jnp.float32),
    }


def init_ffn(key, d, expansion=4):
    k1, k2 = jax.random.split(key)
    hidden = expansion * d
    return {
        "W1": init_linear(k1, d, hidden),
        "b1": jnp.zeros((hidden,), dtype=jnp.float32),
        "W2": init_linear(k2, hidden, d),
        "b2": jnp.zeros((d,), dtype=jnp.float32),
    }


def init_block_params(key, d, gate="gated"):
    kt, ks, kg, kf = jax.random.split(key, 4)
    params = {
        "temporal": init_attention_params(kt, d),
        "spatial": init_attention_params(ks, d),
        "ln1": init_layer_norm(d),
        "ln2": init_layer_norm(d),
        "ffn": init_ffn(kf, d),
    }
    if gate == "fixed":
        params["gate"] = init_fixed_gate(0.0)
    elif gate == "gated":
        params["gate"] = init_token_gate(kg, d)
    elif gate in ("temporal", "spatial"):
        params["gate"] = {}
    else:
        raise ValueError(f"unknown gate {gate!r}")
    return params


def layer_norm(params, x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    xh = (x - mean) / jnp.sqrt(var + eps)
    return params["scale"] * xh + params["bias"]


def ffn(params, x):
    return jax.nn.gelu(x @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]


def aaa_block(
    params,
    x,
    num_heads,
    knn_mask=None,
    tau=1.0,
    path="both",
    return_mix=False,
):
    """Apply one AAA block.

    x: (B, C, T, D). ``path`` is ``"both"`` (gated/fixed mix), ``"temporal"``,
    or ``"spatial"``. Path-only modes are for connectivity tests.
    """
    t_out = temporal_attention(params["temporal"], x, num_heads)
    s_out = spatial_attention(
        params["spatial"], x, num_heads, knn_mask=knn_mask
    )
    gate = params.get("gate") or {}
    if path == "temporal":
        mixed = t_out
    elif path == "spatial":
        mixed = s_out
    elif "ell" in gate:
        mixed = fixed_mix(gate, t_out, s_out)
    elif "W1" in gate:
        mixed = gated_mix(gate, x, t_out, s_out, tau=tau)
    else:
        mixed = 0.5 * t_out + 0.5 * s_out
    h = layer_norm(params["ln1"], x + mixed)
    out = layer_norm(params["ln2"], h + ffn(params["ffn"], h))
    if return_mix:
        return out, mixed, t_out, s_out
    return out
