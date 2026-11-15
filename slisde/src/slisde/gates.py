"""Gated in-flow stacking (Eqs. 7, 8, 19, 81, 82)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


GATE_EPS = 0.1
RMS_S = 1e-6


def rms_norm(z, gamma, s=RMS_S):
    """Π(z) = γ ⊙ z / sqrt(ς + d^{-1} ||z||^2)  (Eq. 19)."""
    d = z.shape[-1]
    scale = jnp.sqrt(s + jnp.mean(z * z, axis=-1, keepdims=True))
    return gamma * z / scale


def time_features(t, omega=2.0 * jnp.pi):
    """τ_gate(t) = (t, t^2, sin(ω t), cos(ω t))  (Eq. 79, gate channel)."""
    t = jnp.asarray(t)
    return jnp.stack([t, t * t, jnp.sin(omega * t), jnp.cos(omega * t)], axis=-1)


def gate_input(z_prev, t, gamma, omega=2.0 * jnp.pi, use_time=True):
    """x_k = (Π(Z_k^{(ℓ-1)}), τ_gate(t_k))  (Eq. 80 without causal conv)."""
    x = rms_norm(z_prev, gamma)
    if use_time:
        tau = time_features(t, omega=omega)
        tau = jnp.broadcast_to(tau, x.shape[:-1] + tau.shape[-1:])
        x = jnp.concatenate([x, tau], axis=-1)
    return x


def init_gates(key, d, x_dim, *, scale=0.01, eps=GATE_EPS):
    """Small-scale GLU kernels so initially α≈1 and o≈0 (Appendix C.4)."""
    keys = jax.random.split(key, 4)
    zeros = jnp.zeros((d,))
    return {
        "W_gf": scale * jax.random.normal(keys[0], (d, x_dim)),
        "b_gf": zeros,
        "W_if": scale * jax.random.normal(keys[1], (d, x_dim)),
        "b_if": zeros,
        "W_go": scale * jax.random.normal(keys[2], (d, x_dim)),
        "b_go": zeros,
        "W_io": scale * jax.random.normal(keys[3], (d, x_dim)),
        "c_io": zeros,
        "gamma": jnp.ones((d,)),
        "eps": jnp.asarray(eps),
    }


def zero_gates(d, x_dim, eps=GATE_EPS):
    """All-zero GLU maps: α=1, o=0 (identity in-flow)."""
    return {
        "W_gf": jnp.zeros((d, x_dim)),
        "b_gf": jnp.zeros((d,)),
        "W_if": jnp.zeros((d, x_dim)),
        "b_if": jnp.zeros((d,)),
        "W_go": jnp.zeros((d, x_dim)),
        "b_go": jnp.zeros((d,)),
        "W_io": jnp.zeros((d, x_dim)),
        "c_io": jnp.zeros((d,)),
        "gamma": jnp.ones((d,)),
        "eps": jnp.asarray(eps),
    }


def _linear(x, W, b):
    return jnp.einsum("...i,ji->...j", x, W) + b


def scale_and_offset(gates, x):
    """Diagonal scale α_k and offset o_k (Eqs. 81–82).

    α_k = 1 + ε σ(W_g^F x_k) ⊙ tanh(W_i^F x_k)  ∈ (1-ε, 1+ε)^d
    o_k = σ(W_go x_k) ⊙ (W_io x_k + c_io)
    Biases on every gate map as in Appendix C.4.
    """
    eps = gates["eps"]
    alpha = 1.0 + eps * jax.nn.sigmoid(_linear(x, gates["W_gf"], gates["b_gf"])) * jnp.tanh(
        _linear(x, gates["W_if"], gates["b_if"])
    )
    offset = jax.nn.sigmoid(_linear(x, gates["W_go"], gates["b_go"])) * (
        _linear(x, gates["W_io"], gates["c_io"])
    )
    return alpha, offset


def scale_diag(F, alpha):
    """D(α, F): rescale the diagonal of F entrywise by α (Eq. 7, Appendix C.4)."""
    d = F.shape[-1]
    eye = jnp.eye(d, dtype=F.dtype)
    new_diag = alpha * jnp.diagonal(F, axis1=-2, axis2=-1)
    return F * (1.0 - eye) + new_diag[..., None] * eye


def apply_inflow(F, g, alpha, offset):
    """F̄_k = D(α_k, F_k),  ḡ_k = g_k + o_k  (Eq. 7)."""
    return scale_diag(F, alpha), g + offset


def gated_pairs(gates, F, g, z_prev_left, t_left, *, omega=2.0 * jnp.pi, use_time=True):
    """Modulate affine pairs by the previous-layer path at left endpoints."""
    x = gate_input(z_prev_left, t_left, gates["gamma"], omega=omega, use_time=use_time)
    alpha, offset = scale_and_offset(gates, x)
    return apply_inflow(F, g, alpha, offset)
