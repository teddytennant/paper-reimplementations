"""Transformer encoder TQSC (SM §§III, VII, VIII)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from jax_tqsc.physics import EPS, rho_from_chol

Params = dict[str, Any]


@dataclass(frozen=True)
class TQSCConfig:
    n_tokens: int = 13
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 256
    eps: float = EPS
    ln_eps: float = 1e-5


def _linear(key: jax.Array, d_in: int, d_out: int) -> dict[str, jax.Array]:
    w = jax.random.normal(key, (d_in, d_out)) * (1.0 / (d_in ** 0.5))
    return {"W": w, "b": jnp.zeros((d_out,))}


def _layernorm_params(d: int) -> dict[str, jax.Array]:
    return {"gamma": jnp.ones((d,)), "beta": jnp.zeros((d,))}


def _init_encoder_layer(key: jax.Array, cfg: TQSCConfig) -> dict[str, Any]:
    d = cfg.d_model
    kq, kk, kv, ko, k1, k2 = jax.random.split(key, 6)
    return {
        "Wq": _linear(kq, d, d),
        "Wk": _linear(kk, d, d),
        "Wv": _linear(kv, d, d),
        "Wo": _linear(ko, d, d),
        "ln1": _layernorm_params(d),
        "ff1": _linear(k1, d, cfg.d_ff),
        "ff2": _linear(k2, cfg.d_ff, d),
        "ln2": _layernorm_params(d),
    }


def init_tqsc(key: jax.Array, cfg: TQSCConfig | None = None) -> Params:
    cfg = cfg or TQSCConfig()
    k_emb, k_head, k_layers = jax.random.split(key, 3)
    layer_keys = jax.random.split(k_layers, cfg.n_layers)
    return {
        "embed": _linear(k_emb, 1, cfg.d_model),
        "layers": [_init_encoder_layer(k, cfg) for k in layer_keys],
        "head": _linear(k_head, cfg.d_model, 4),
    }


def sinusoidal_positional_encoding(length: int, d_model: int) -> jax.Array:
    pos = jnp.arange(length, dtype=jnp.float32)[:, None]
    dim = jnp.arange(d_model, dtype=jnp.float32)[None, :]
    angle = pos / (10000.0 ** (2.0 * (dim // 2) / d_model))
    even = (dim.astype(jnp.int32) % 2) == 0
    return jnp.where(even, jnp.sin(angle), jnp.cos(angle))


def _layernorm(x: jax.Array, p: dict[str, jax.Array], eps: float) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    xhat = (x - mean) / jnp.sqrt(var + eps)
    return p["gamma"] * xhat + p["beta"]


def _dense(p: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    return x @ p["W"] + p["b"]


def _mhsa(
    layer: dict[str, Any],
    x: jax.Array,
    n_heads: int,
) -> tuple[jax.Array, jax.Array]:
    """Scaled dot-product multi-head attention (S39–S44). Returns (out, weights)."""
    b, length, d_model = x.shape
    d_k = d_model // n_heads
    q = _dense(layer["Wq"], x).reshape(b, length, n_heads, d_k).transpose(0, 2, 1, 3)
    k = _dense(layer["Wk"], x).reshape(b, length, n_heads, d_k).transpose(0, 2, 1, 3)
    v = _dense(layer["Wv"], x).reshape(b, length, n_heads, d_k).transpose(0, 2, 1, 3)
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.asarray(d_k, dtype=x.dtype))
    weights = jax.nn.softmax(scores, axis=-1)
    heads = jnp.matmul(weights, v)
    concat = heads.transpose(0, 2, 1, 3).reshape(b, length, d_model)
    return _dense(layer["Wo"], concat), weights


def _encoder_layer(
    layer: dict[str, Any],
    x: jax.Array,
    n_heads: int,
    ln_eps: float,
) -> tuple[jax.Array, jax.Array]:
    attn, weights = _mhsa(layer, x, n_heads)
    x = _layernorm(x + attn, layer["ln1"], ln_eps)
    h = jax.nn.gelu(_dense(layer["ff1"], x))
    ff = _dense(layer["ff2"], h)
    x = _layernorm(x + ff, layer["ln2"], ln_eps)
    return x, weights


def _infer_heads(d_model: int) -> int:
    if d_model % 16 == 0:
        return d_model // 16
    if d_model % 4 == 0:
        return 4
    return 1


def forward(
    params: Params,
    measurements: jax.Array,
    return_attn: bool = False,
    cfg: TQSCConfig | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Map (B, 13) real measurements to (B, 2, 2) complex density matrices."""
    m = jnp.asarray(measurements, dtype=jnp.float32)
    tokens = m[..., :, None]
    h = _dense(params["embed"], tokens)
    d_model = h.shape[-1]
    n_heads = cfg.n_heads if cfg is not None else _infer_heads(d_model)
    ln_eps = cfg.ln_eps if cfg is not None else 1e-5
    chol_eps = cfg.eps if cfg is not None else EPS
    pe = sinusoidal_positional_encoding(m.shape[1], d_model).astype(h.dtype)
    h = h + pe
    attn = None
    for layer in params["layers"]:
        h, attn = _encoder_layer(layer, h, n_heads, ln_eps)
    h_agg = jnp.mean(h, axis=-2)
    r = _dense(params["head"], h_agg)
    rho = rho_from_chol(r, eps=chol_eps)
    if return_attn:
        return rho, attn
    return rho


def count_params(params: Params) -> int:
    leaves = [p for p in jax.tree.leaves(params) if isinstance(p, jax.Array)]
    return int(sum(p.size for p in leaves))
