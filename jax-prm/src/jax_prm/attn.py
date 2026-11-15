"""Causal self-attention, memory cross-attention, LayerNorm, linear init."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

Array = jax.Array
Params = dict[str, Any]


def init_linear(key: Array, in_dim: int, out_dim: int, bias: bool = True) -> Params:
    """Lecun-normal (fan-in) linear. Weight shape (in_dim, out_dim)."""
    std = in_dim**-0.5
    params: Params = {"W": jax.random.normal(key, (in_dim, out_dim)) * std}
    if bias:
        params["b"] = jnp.zeros((out_dim,), dtype=jnp.float32)
    return params


def linear(params: Params, x: Array) -> Array:
    y = x @ params["W"]
    if "b" in params:
        y = y + params["b"]
    return y


def init_layer_norm(dim: int) -> Params:
    return {"scale": jnp.ones((dim,), dtype=jnp.float32), "bias": jnp.zeros((dim,), dtype=jnp.float32)}


def layer_norm(params: Params, x: Array, eps: float = 1e-5) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return params["scale"] * (x - mean) * jax.lax.rsqrt(var + eps) + params["bias"]


def dropout(x: Array, rate: float, key: Array | None) -> Array:
    if rate is None or rate <= 0.0 or key is None:
        return x
    keep = 1.0 - rate
    mask = jax.random.bernoulli(key, keep, x.shape).astype(x.dtype)
    return x * mask / keep


def causal_mask(t: int, dtype=jnp.float32) -> Array:
    """Additive mask (T, T): 0 on/below diagonal, large negative above."""
    keep = jnp.tril(jnp.ones((t, t), dtype=bool))
    return jnp.where(keep, jnp.array(0.0, dtype=dtype), jnp.array(-1e9, dtype=dtype))


def split_heads(x: Array, n_heads: int) -> Array:
    """(B, T, D) -> (B, H, T, d_h)."""
    b, t, d = x.shape
    d_h = d // n_heads
    return x.reshape(b, t, n_heads, d_h).transpose(0, 2, 1, 3)


def merge_heads(x: Array) -> Array:
    """(B, H, T, d_h) -> (B, T, D)."""
    b, n_heads, t, d_h = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, n_heads * d_h)


def init_self_attn(key: Array, d_model: int) -> Params:
    kq, kk, kv, ko = jax.random.split(key, 4)
    return {
        "q": init_linear(kq, d_model, d_model, bias=True),
        "k": init_linear(kk, d_model, d_model, bias=True),
        "v": init_linear(kv, d_model, d_model, bias=True),
        "o": init_linear(ko, d_model, d_model, bias=True),
    }


def self_attention(params: Params, x: Array, n_heads: int) -> tuple[Array, Array]:
    """Causal multi-head self-attention.

    Returns (out, weights) with out (B, T, D) and weights (B, H, T, T).
    Position i cannot attend to j > i: those weights are ~0.
    """
    q = split_heads(linear(params["q"], x), n_heads)
    k = split_heads(linear(params["k"], x), n_heads)
    v = split_heads(linear(params["v"], x), n_heads)
    d_h = q.shape[-1]
    scores = jnp.matmul(q, k.swapaxes(-1, -2)) * (d_h**-0.5)
    t = x.shape[1]
    scores = scores + causal_mask(t, dtype=scores.dtype)
    weights = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.matmul(weights, v)
    return linear(params["o"], merge_heads(ctx)), weights


def init_cross_attn(key: Array, d_mem: int, d_model: int) -> Params:
    """Queries from memory (d_mem), keys/values from H (d_model); out back to d_mem."""
    kq, kk, kv, ko = jax.random.split(key, 4)
    return {
        "Wq": init_linear(kq, d_mem, d_model, bias=False)["W"],
        "Wk": init_linear(kk, d_model, d_model, bias=False)["W"],
        "Wv": init_linear(kv, d_model, d_model, bias=False)["W"],
        "Wo": init_linear(ko, d_model, d_mem, bias=False)["W"],
    }


def cross_attend_memory(
    params: Params,
    memory: Array,
    h: Array,
    n_heads: int,
    t_index: Array | None = None,
) -> Array:
    """M' = CrossAttn(Q=M, K=H, V=H) with an optional causal cutoff at t_index.

    memory: (B, S, d_mem), h: (B, T, d_model).
    If t_index is given (scalar), keys at positions > t_index are masked so a
    slot update at token t only sees H_{<=t}.
    Output is projected back to d_mem: (B, S, d_mem).
    """
    q = memory @ params["Wq"]
    k = h @ params["Wk"]
    v = h @ params["Wv"]
    qh = split_heads(q, n_heads)
    kh = split_heads(k, n_heads)
    vh = split_heads(v, n_heads)
    d_h = qh.shape[-1]
    scores = jnp.matmul(qh, kh.swapaxes(-1, -2)) * (d_h**-0.5)
    if t_index is not None:
        seq = h.shape[1]
        allowed = jnp.arange(seq) <= t_index
        scores = jnp.where(allowed[None, None, None, :], scores, jnp.array(-1e9, dtype=scores.dtype))
    weights = jax.nn.softmax(scores, axis=-1)
    ctx = merge_heads(jnp.matmul(weights, vh))
    return ctx @ params["Wo"]
