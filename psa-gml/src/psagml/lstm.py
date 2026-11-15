"""Coordinate-wise LSTM optimizer (eqs 29-30) and gradient preprocess."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class LSTMParams(NamedTuple):
    Wf: jnp.ndarray
    Wi: jnp.ndarray
    Wc: jnp.ndarray
    Wo: jnp.ndarray
    Uf: jnp.ndarray
    Ui: jnp.ndarray
    Uc: jnp.ndarray
    Uo: jnp.ndarray
    bf: jnp.ndarray
    bi: jnp.ndarray
    bc: jnp.ndarray
    bo: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray


def preprocess_grad(g):
    """Eq (29): g_d = sgn(grad) log(1 + |grad|). Odd, and 0 maps to 0."""
    return jnp.sign(g) * jnp.log1p(jnp.abs(g))


def init_lstm(key, hidden, scale=0.1):
    """Random LSTM cell: W in R^{H}, U in R^{H x H}, readout (a, b)."""
    keys = jax.random.split(key, 10)
    h = hidden
    u_scale = 1.0 / jnp.sqrt(jnp.asarray(h, dtype=jnp.float32))

    def wvec(k):
        return jax.random.normal(k, (h,)) * scale

    def umat(k):
        return jax.random.normal(k, (h, h)) * u_scale

    return LSTMParams(
        Wf=wvec(keys[0]),
        Wi=wvec(keys[1]),
        Wc=wvec(keys[2]),
        Wo=wvec(keys[3]),
        Uf=umat(keys[4]),
        Ui=umat(keys[5]),
        Uc=umat(keys[6]),
        Uo=umat(keys[7]),
        bf=jnp.zeros((h,)),
        bi=jnp.zeros((h,)),
        bc=jnp.zeros((h,)),
        bo=jnp.zeros((h,)),
        a=wvec(keys[8]),
        b=jnp.zeros(()),
    )


def lstm_step(params: LSTMParams, g, h, c, nu):
    """One coordinate-wise LSTM step (eq 30).

    g: (D,), h and c: (D, H). Returns (h_new, c_new, delta) with delta (D,).
    """
    # W_* g is (H,) per coordinate: g[:, None] * W[None, :]
    def affine(W, U, b):
        return g[:, None] * W + h @ U.T + b

    f = jax.nn.sigmoid(affine(params.Wf, params.Uf, params.bf))
    i = jax.nn.sigmoid(affine(params.Wi, params.Ui, params.bi))
    c_tilde = jnp.tanh(affine(params.Wc, params.Uc, params.bc))
    o = jax.nn.sigmoid(affine(params.Wo, params.Uo, params.bo))
    c_new = f * c + i * c_tilde
    h_new = o * jnp.tanh(c_new)
    delta = nu * (h_new @ params.a + params.b)
    return h_new, c_new, delta


def init_lstm_state(n_coords, hidden, dtype=jnp.float32):
    zeros = jnp.zeros((n_coords, hidden), dtype=dtype)
    return zeros, zeros
