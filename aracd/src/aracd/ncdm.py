"""NCDM backbone used as F_CD (paper eqs. 8–9)."""

from __future__ import annotations

import jax.numpy as jnp

from aracd.ops import sigmoid


def ncdm_interaction(e_s, d_disc, d_kc, q_i):
    """Eq. 8: x_si = d_i^{disc} ⊙ (e_s - d_i^{kc}) ⊙ q_i.

    All arguments share a trailing concept axis of size K. A leading batch
    axis is optional. Entries with q_ik = 0 are zeroed.
    """
    return d_disc * (e_s - d_kc) * q_i


def ncdm_mlp(x_si, w1, b1, w2, b2):
    """Tiny MLP: Linear → sigmoid → Linear → scalar (pre-sigmoid).

    x_si: (..., K)
    w1: (K, H), b1: (H,)
    w2: (H,), b2: ()
    returns: (...)
    """
    hidden = sigmoid(x_si @ w1 + b1)
    return hidden @ w2 + b2


def ncdm_base(e_s, d_disc, d_kc, q_i, w1, b1, w2, b2):
    """Eqs. 8–9: y_si^{base} = σ(MLP(x_si)). Returns (y_base, x_si)."""
    x_si = ncdm_interaction(e_s, d_disc, d_kc, q_i)
    y_base = sigmoid(ncdm_mlp(x_si, w1, b1, w2, b2))
    return y_base, x_si
