"""Tiny CavityNet: 8x8 geometry to tokens (stand-in for the paper's 18x18 CNN)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_nmd.config import NMDConfig
from jax_nmd.poles import _layernorm


def _init_conv(key, kh, kw, cin, cout):
    fan_in = kh * kw * cin
    w = jax.random.normal(key, (kh, kw, cin, cout), dtype=jnp.float32) * (fan_in ** -0.5)
    b = jnp.zeros((cout,), dtype=jnp.float32)
    return w, b


def _conv(x, w, b, stride=1):
    y = jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + b


def init_cavity_net(key, cfg: NMDConfig):
    k1, k2, kpos = jax.random.split(key, 3)
    w1, b1 = _init_conv(k1, 3, 3, 1, cfg.conv_mid)
    w2, b2 = _init_conv(k2, 3, 3, cfg.conv_mid, cfg.d_t)
    pos = jax.random.normal(kpos, (cfg.n_tokens, cfg.d_t), dtype=jnp.float32) * 0.02
    return {
        "conv1_w": w1,
        "conv1_b": b1,
        "conv2_w": w2,
        "conv2_b": b2,
        "pos": pos,
        "ln_g": jnp.ones((cfg.d_t,), dtype=jnp.float32),
        "ln_b": jnp.zeros((cfg.d_t,), dtype=jnp.float32),
    }


def cavity_encode(params, geometry, cfg: NMDConfig):
    """geometry (B, H, W) binary/float -> tokens (B, n_tokens, D_t)."""
    x = jnp.asarray(geometry, dtype=jnp.float32)
    if x.ndim == 3:
        x = x[..., None]
    h = jax.nn.gelu(_conv(x, params["conv1_w"], params["conv1_b"], stride=2))
    h = jax.nn.gelu(_conv(h, params["conv2_w"], params["conv2_b"], stride=2))
    tokens = h.reshape(h.shape[0], -1, h.shape[-1])
    tokens = tokens + params["pos"]
    return _layernorm(tokens, params["ln_g"], params["ln_b"], eps=cfg.ln_eps)
