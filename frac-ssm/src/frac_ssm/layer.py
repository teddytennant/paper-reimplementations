"""Tiny FracMixer and FRAC layer (Section 3.3).

RMSNorm, linear in, FracMixer (control Δ/α/λ, depthwise causal conv content,
Alg 1 transition, feedthrough D u), gated out. Widths: d=16, M=4, H=2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .bank import geometric_timescales
from .transition import frac_step, gated_mode_logits, transition_sequence

DEFAULT_D = 16
DEFAULT_H = 2
DEFAULT_M = 4
DEFAULT_CONV_K = 3
DEFAULT_TAU0 = 0.5
DEFAULT_Q = 2.0


def rms_norm(x, weight, eps=1e-6):
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x / rms * weight


def causal_dw_conv1d(x, kernel):
    """Depthwise causal conv1d, kernel size K.

    x: (B, T, D), kernel: (K, D) -> (B, T, D)
    """
    k = kernel.shape[0]
    x_pad = jnp.pad(x, ((0, 0), (k - 1, 0), (0, 0)))
    lhs = jnp.transpose(x_pad, (0, 2, 1))
    rhs = jnp.transpose(kernel, (1, 0))[:, None, :]
    out = jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCW", "OIW", "NCW"),
        feature_group_count=x.shape[-1],
    )
    return jnp.transpose(out, (0, 2, 1))


def conv_step(x_t, conv_state, kernel):
    """One causal conv step. x_t: (B, D), conv_state: (B, K-1, D)."""
    window = jnp.concatenate([conv_state, x_t[:, None, :]], axis=1)
    y = jnp.sum(window * kernel[None, :, :], axis=1)
    new_state = window[:, 1:, :]
    return y, new_state


def _split_heads(x, n_heads):
    B, T, D = x.shape
    dh = D // n_heads
    return x.reshape(B, T, n_heads, dh)


def _merge_heads(x):
    B, T, H, dh = x.shape
    return x.reshape(B, T, H * dh)


def init_frac_mixer(key, d, n_heads, n_modes, conv_k=DEFAULT_CONV_K):
    dh = d // n_heads
    k_delta, k_alpha, k_lam, k_conv, k_read, k_write, k_d = jax.random.split(key, 7)
    scale = 0.02
    return {
        "W_delta": scale * jax.random.normal(k_delta, (d, n_heads)),
        "b_delta": jnp.full((n_heads,), 0.5),
        "W_alpha": scale * jax.random.normal(k_alpha, (d, n_heads)),
        "b_alpha": jnp.zeros((n_heads,)),
        "W_lambda": scale * jax.random.normal(k_lam, (d, n_heads)),
        "b_lambda": jnp.zeros((n_heads,)),
        "conv": (1.0 / conv_k) * jax.random.normal(k_conv, (conv_k, d)),
        "W_read": scale * jax.random.normal(k_read, (n_heads, n_modes, dh)),
        "p_read": jnp.array(0.0),
        "W_write": scale * jax.random.normal(k_write, (n_heads, n_modes, dh)),
        "p_write": jnp.array(0.0),
        "D_feed": jax.random.normal(k_d, (n_heads, dh)) * 0.1,
    }


def init_frac_layer(
    key,
    d=DEFAULT_D,
    n_heads=DEFAULT_H,
    n_modes=DEFAULT_M,
    conv_k=DEFAULT_CONV_K,
    tau0=DEFAULT_TAU0,
    q=DEFAULT_Q,
):
    k_in, k_gate, k_out, k_mix = jax.random.split(key, 4)
    scale = 0.02
    return {
        "norm_w": jnp.ones((d,)),
        "in_w": scale * jax.random.normal(k_in, (d, d)),
        "in_b": jnp.zeros((d,)),
        "gate_w": scale * jax.random.normal(k_gate, (d, d)),
        "gate_b": jnp.zeros((d,)),
        "out_w": scale * jax.random.normal(k_out, (d, d)),
        "out_b": jnp.zeros((d,)),
        "mixer": init_frac_mixer(k_mix, d, n_heads, n_modes, conv_k=conv_k),
        "tau": geometric_timescales(tau0, q, n_modes),
    }


def mixer_controls(mixer, x):
    """Δ=softplus(WΔ x + b), α=sigmoid(Wα x), λ=softplus(Wλ x). x: (B, T, D)."""
    delta = jnp.clip(jax.nn.softplus(x @ mixer["W_delta"] + mixer["b_delta"]), 1e-4, 10.0)
    alpha = jnp.clip(jax.nn.sigmoid(x @ mixer["W_alpha"] + mixer["b_alpha"]), 0.05, 0.99)
    lam = jnp.clip(jax.nn.softplus(x @ mixer["W_lambda"] + mixer["b_lambda"]), 1e-3, 50.0)
    return delta, alpha, lam


def apply_mixer(mixer, x, tau, s0=None, parallel=True):
    """FracMixer over a full sequence.

    x: (B, T, D). Returns y: (B, T, D), S_last: (B, H, M, dh)
    """
    n_heads = mixer["W_read"].shape[0]
    delta, alpha, lam = mixer_controls(mixer, x)
    u = causal_dw_conv1d(x, mixer["conv"])
    u_h = _split_heads(u, n_heads)
    g_read = gated_mode_logits(u_h, mixer["W_read"], mixer["p_read"])
    g_write = gated_mode_logits(u_h, mixer["W_write"], mixer["p_write"])
    h, S = transition_sequence(
        u_h, delta, alpha, lam, tau, g_read, g_write, s0=s0, parallel=parallel
    )
    y = h + mixer["D_feed"] * u_h
    return _merge_heads(y), S[:, -1]


def apply_mixer_step(mixer, x_t, tau, cache):
    """One-token mixer step with cached conv and mode state.

    x_t: (B, D). cache: dict with S (B,H,M,dh) and conv (B, K-1, D).
    """
    n_heads = mixer["W_read"].shape[0]
    x_seq = x_t[:, None, :]
    delta, alpha, lam = mixer_controls(mixer, x_seq)
    u_t, conv_state = conv_step(x_t, cache["conv"], mixer["conv"])
    u_h = u_t.reshape(u_t.shape[0], n_heads, -1)
    g_read = gated_mode_logits(u_h, mixer["W_read"], mixer["p_read"])
    g_write = gated_mode_logits(u_h, mixer["W_write"], mixer["p_write"])
    h, S = frac_step(
        u_h,
        delta[:, 0],
        alpha[:, 0],
        lam[:, 0],
        cache["S"],
        tau,
        g_read,
        g_write,
    )
    y = h + mixer["D_feed"] * u_h
    y = y.reshape(y.shape[0], -1)
    return y, {"S": S, "conv": conv_state}


def apply_layer(params, x, parallel=True, cache=None, return_cache=False):
    """FRAC layer: RMSNorm, linear in, mixer, gated out, residual.

    x: (B, T, D) -> y: (B, T, D). State is (B, H, M, dh).
    """
    tau = jax.lax.stop_gradient(params["tau"])
    residual = x
    n = rms_norm(x, params["norm_w"])
    mixer_in = n @ params["in_w"] + params["in_b"]
    gate = n @ params["gate_w"] + params["gate_b"]
    s0 = None if cache is None else cache["S"]
    y, S_last = apply_mixer(
        params["mixer"], mixer_in, tau, s0=s0, parallel=parallel
    )
    y = y * jax.nn.silu(gate)
    y = y @ params["out_w"] + params["out_b"]
    out = residual + y
    if return_cache:
        k = params["mixer"]["conv"].shape[0]
        conv_cache = mixer_in[:, -(k - 1) :, :]
        if conv_cache.shape[1] < k - 1:
            conv_cache = jnp.pad(
                conv_cache, ((0, 0), (k - 1 - conv_cache.shape[1], 0), (0, 0))
            )
        return out, {"S": S_last, "conv": conv_cache}
    return out, S_last


def empty_cache(batch, d, n_heads, n_modes, conv_k=DEFAULT_CONV_K, dtype=jnp.float32):
    dh = d // n_heads
    return {
        "S": jnp.zeros((batch, n_heads, n_modes, dh), dtype=dtype),
        "conv": jnp.zeros((batch, conv_k - 1, d), dtype=dtype),
    }


def apply_layer_step(params, x_t, cache):
    """Autoregressive one-token step. x_t: (B, D)."""
    tau = jax.lax.stop_gradient(params["tau"])
    n = rms_norm(x_t, params["norm_w"])
    mixer_in = n @ params["in_w"] + params["in_b"]
    gate = n @ params["gate_w"] + params["gate_b"]
    y, cache = apply_mixer_step(params["mixer"], mixer_in, tau, cache)
    y = y * jax.nn.silu(gate)
    y = y @ params["out_w"] + params["out_b"]
    return x_t + y, cache


def decode_autoregressive(params, x):
    """Token-by-token decoding with cached state, same as parallel prefill."""
    B, T, D = x.shape
    n_heads = params["mixer"]["W_read"].shape[0]
    n_modes = params["mixer"]["W_read"].shape[1]
    cache = empty_cache(B, D, n_heads, n_modes, params["mixer"]["conv"].shape[0], x.dtype)

    def step(cache, x_t):
        y_t, cache = apply_layer_step(params, x_t, cache)
        return cache, y_t

    _, ys = jax.lax.scan(step, cache, jnp.swapaxes(x, 0, 1))
    return jnp.swapaxes(ys, 0, 1)
