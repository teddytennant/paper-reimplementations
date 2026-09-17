"""Pole parameterization (eq 3, appendix B.2 eq 10) and PoleNet."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_nmd.config import NMDConfig


def poles_from_raw(u, v):
    """p_k = -γ_k + j ω_k with γ_k = softplus(u_k), ω_k = v_k.

    All poles lie in the open left half-plane because softplus(u) > 0.
    """
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    gamma = jax.nn.softplus(u)
    return jax.lax.complex(-gamma, v)


def _linear(x, w, b):
    return x @ w + b


def _layernorm(x, gamma, beta, eps=1e-5):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return gamma * (x - mu) * jax.lax.rsqrt(var + eps) + beta


def _cross_attn(q, kv, wq, wk, wv, wo):
    qd = q @ wq
    kd = kv @ wk
    vd = kv @ wv
    scale = jax.lax.rsqrt(jnp.asarray(qd.shape[-1], dtype=qd.dtype))
    logits = jnp.matmul(qd, jnp.swapaxes(kd, -1, -2)) * scale
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.matmul(weights, vd) @ wo


def _ffn(x, w1, b1, w2, b2):
    return _linear(jax.nn.gelu(_linear(x, w1, b1)), w2, b2)


def _init_linear(key, din, dout, scale=None):
    if scale is None:
        scale = (2.0 / (din + dout)) ** 0.5
    w = jax.random.normal(key, (din, dout), dtype=jnp.float32) * scale
    b = jnp.zeros((dout,), dtype=jnp.float32)
    return w, b


def init_attn_block(key, d_t, hidden):
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    wq, _ = _init_linear(k1, d_t, d_t)
    wk, _ = _init_linear(k2, d_t, d_t)
    wv, _ = _init_linear(k3, d_t, d_t)
    wo, _ = _init_linear(k4, d_t, d_t)
    w1, b1 = _init_linear(k5, d_t, hidden)
    w2, b2 = _init_linear(k6, hidden, d_t)
    return {
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "wo": wo,
        "ff1_w": w1,
        "ff1_b": b1,
        "ff2_w": w2,
        "ff2_b": b2,
        "ln_g": jnp.ones((d_t,), dtype=jnp.float32),
        "ln_b": jnp.zeros((d_t,), dtype=jnp.float32),
    }


def attn_block(params, q, kv, ln_eps=1e-5):
    x = _layernorm(
        q + _cross_attn(q, kv, params["wq"], params["wk"], params["wv"], params["wo"]),
        params["ln_g"],
        params["ln_b"],
        eps=ln_eps,
    )
    return x + _ffn(x, params["ff1_w"], params["ff1_b"], params["ff2_w"], params["ff2_b"])


def init_pole_net(key, cfg: NMDConfig):
    k_q, k_attn, k_out = jax.random.split(key, 3)
    d_t, k_poles = cfg.d_t, cfg.k_poles
    queries = jax.random.normal(k_q, (k_poles, d_t), dtype=jnp.float32) * 0.02
    out_w, _ = _init_linear(k_out, d_t, 2, scale=0.02)
    u_bias = jnp.zeros((k_poles,), dtype=jnp.float32)
    v_bias = jnp.linspace(-3.0, 3.0, k_poles, dtype=jnp.float32)
    out_b = jnp.stack([u_bias, v_bias], axis=-1)
    return {
        "queries": queries,
        "attn": init_attn_block(k_attn, d_t, cfg.hidden),
        "out_w": out_w,
        "out_b": out_b,
    }


def pole_net(params, tokens, cfg: NMDConfig):
    """tokens (B, n_tokens, D_t) -> poles (B, K) complex."""
    queries = params["queries"]
    q = jnp.broadcast_to(queries, (tokens.shape[0],) + queries.shape)
    x = attn_block(params["attn"], q, tokens, ln_eps=cfg.ln_eps)
    uv = x @ params["out_w"] + params["out_b"]
    return poles_from_raw(uv[..., 0], uv[..., 1])
