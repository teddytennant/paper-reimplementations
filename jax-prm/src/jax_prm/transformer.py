"""Decoder-only transformer with PRM inserted after the lower half of the stack.

Pre-LN blocks, causal self-attention, GELU FFN, learned position embeddings,
untied input/output embeddings. PRM runs after layer n_layers//2.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jax_prm.attn import (
    dropout,
    init_layer_norm,
    init_linear,
    init_self_attn,
    layer_norm,
    linear,
    self_attention,
)
from jax_prm.config import PRMConfig
from jax_prm.prm import init_prm, prm_forward

Array = jax.Array
Params = dict[str, Any]


def init_ffn(key: Array, d_model: int, d_ff: int) -> Params:
    k1, k2 = jax.random.split(key)
    return {
        "in": init_linear(k1, d_model, d_ff, bias=True),
        "out": init_linear(k2, d_ff, d_model, bias=True),
    }


def ffn(params: Params, x: Array) -> Array:
    return linear(params["out"], jax.nn.gelu(linear(params["in"], x)))


def init_block(key: Array, cfg: PRMConfig) -> Params:
    ka, kf = jax.random.split(key)
    return {
        "ln1": init_layer_norm(cfg.d_model),
        "attn": init_self_attn(ka, cfg.d_model),
        "ln2": init_layer_norm(cfg.d_model),
        "ffn": init_ffn(kf, cfg.d_model, cfg.d_ff),
    }


def transformer_block(
    params: Params,
    x: Array,
    cfg: PRMConfig,
    *,
    key: Array | None = None,
    training: bool = False,
) -> tuple[Array, Array]:
    y, weights = self_attention(params["attn"], layer_norm(params["ln1"], x, eps=cfg.ln_eps), cfg.n_heads)
    if training and cfg.dropout > 0.0 and key is not None:
        key, dk = jax.random.split(key)
        y = dropout(y, cfg.dropout, dk)
    x = x + y
    y = ffn(params["ffn"], layer_norm(params["ln2"], x, eps=cfg.ln_eps))
    if training and cfg.dropout > 0.0 and key is not None:
        y = dropout(y, cfg.dropout, key)
    return x + y, weights


def init(key: Array, cfg: PRMConfig) -> Params:
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(f"d_model={cfg.d_model} not divisible by n_heads={cfg.n_heads}")
    if cfg.prm_enabled and cfg.d_model % cfg.n_mem_heads != 0:
        raise ValueError(f"d_model={cfg.d_model} not divisible by n_mem_heads={cfg.n_mem_heads}")

    kt, kp, ku, kl, kprm = jax.random.split(key, 5)
    layer_keys = jax.random.split(kl, cfg.n_layers)
    params: Params = {
        "tok_emb": jax.random.normal(kt, (cfg.vocab_size, cfg.d_model)) * 0.02,
        "pos_emb": jax.random.normal(kp, (cfg.context, cfg.d_model)) * 0.02,
        "layers": [init_block(layer_keys[i], cfg) for i in range(cfg.n_layers)],
        "ln_f": init_layer_norm(cfg.d_model),
        "unembed": jax.random.normal(ku, (cfg.d_model, cfg.vocab_size)) * (cfg.d_model**-0.5),
    }
    if cfg.prm_enabled:
        params["prm"] = init_prm(kprm, cfg)
    return params


def count_params(params: Params) -> int:
    return int(sum(int(x.size) for x in jax.tree_util.tree_leaves(params)))


def forward(
    params: Params,
    tokens: Array,
    cfg: PRMConfig,
    *,
    key: Array | None = None,
    training: bool = False,
) -> tuple[Array, dict[str, Any]]:
    """tokens: (B, T) int ids. Returns logits (B, T, vocab) and an info dict."""
    _b, t_len = tokens.shape
    if t_len > cfg.context:
        raise ValueError(f"sequence length {t_len} exceeds context {cfg.context}")
    x = params["tok_emb"][tokens] + params["pos_emb"][:t_len]
    if training and cfg.dropout > 0.0 and key is not None:
        key, dk = jax.random.split(key)
        x = dropout(x, cfg.dropout, dk)

    attn_weights = None
    h_lower = None
    h_prm = None
    memory = None
    split_after = cfg.split_after

    for i in range(cfg.n_layers):
        if key is not None:
            key, bk = jax.random.split(key)
        else:
            bk = None
        x, weights = transformer_block(params["layers"][i], x, cfg, key=bk, training=training)
        if i == 0:
            attn_weights = weights
        if i == split_after:
            h_lower = x
            if cfg.prm_enabled:
                if key is not None:
                    key, pk = jax.random.split(key)
                else:
                    pk = None
                x, memory = prm_forward(params["prm"], x, cfg, key=pk, training=training)
                h_prm = x

    h = layer_norm(params["ln_f"], x, eps=cfg.ln_eps)
    logits = h @ params["unembed"]
    info = {
        "h": h,
        "h_lower": h_lower,
        "h_prm": h_prm if h_prm is not None else h_lower,
        "memory": memory,
        "attn_weights": attn_weights,
    }
    return logits, info
