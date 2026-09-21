"""Affective module: student–item affect and guess/slip (eqs. 23–26)."""

from __future__ import annotations

import jax.numpy as jnp

from aracd.ops import sigmoid


def affect_mlp(a_s, d_kc, w1, b1, w2, b2):
    """Eq. 23: a_si = g_θ(a_s, d_i^{kc}) as a small MLP on concat(a_s, d_kc).

    a_s: (..., d_aff), d_kc: (..., K)
    w1: (d_aff + K, H), b1: (H,)
    w2: (H, d_out), b2: (d_out,)
    returns a_si: (..., d_out)
    """
    z = jnp.concatenate([a_s, d_kc], axis=-1)
    hidden = sigmoid(z @ w1 + b1)
    return hidden @ w2 + b2


def guess_slip(a_si, w_g, b_g, w_slip, b_slip):
    """Eqs. 24–25: g_si = σ(w_g · a_si + b_g), s_si = σ(w_slip · a_si + b_slip).

    a_si: (..., d_out)
    w_g, w_slip: (d_out,)
    b_g, b_slip: ()
    returns g, s with shape (...)
    """
    g = sigmoid(a_si @ w_g + b_g)
    s = sigmoid(a_si @ w_slip + b_slip)
    return g, s


def modulate(y_star, g, s):
    """Eq. 26: ŷ = (1 − s) y* + g (1 − y*)."""
    return (1.0 - s) * y_star + g * (1.0 - y_star)
