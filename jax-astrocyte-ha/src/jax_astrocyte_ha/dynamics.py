"""Two-timescale N-RNN / A-RNN dynamics (Eqs. 1-5)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_astrocyte_ha.weff import effective_weights


class NRNNParams(NamedTuple):
    J_xx: jnp.ndarray  # (Nx, Nx)
    H_px: jnp.ndarray  # (Nx, M)
    B_dx: jnp.ndarray  # (Nx, Nu)
    W_qx: jnp.ndarray  # (Ny, Nx)


class ARNNParams(NamedTuple):
    W_hh: jnp.ndarray  # (Nh, Nh)
    W_oh: jnp.ndarray  # (Nh, No)
    b_h: jnp.ndarray  # (Nh,)
    W1: jnp.ndarray  # (Hhid, Nh)
    b1: jnp.ndarray  # (Hhid,)
    W2: jnp.ndarray  # (M, Hhid)
    b2: jnp.ndarray  # (M,)
    W_v: jnp.ndarray  # (Nh,)
    b_v: jnp.ndarray  # ()


def decode(x, W_qx):
    """Linear readout y = W_qx x (Eq. 5)."""
    return W_qx @ x


def nrnn_step(x, u, pi, params: NRNNParams, tau_x, eps=0.0):
    """One fast intra-trial N-RNN step (Eq. 1).

    x_{t+1} = (1-τ_x) x_t + τ_x [ W_eff(π) tanh(x_t) + B_dx u_t + ε_x ]

    If τ_x = 0 the state is unchanged.
    """
    weff = effective_weights(params.J_xx, params.H_px, pi)
    drive = weff @ jnp.tanh(x) + params.B_dx @ u + eps
    return (1.0 - tau_x) * x + tau_x * drive


def unroll_nrnn(params: NRNNParams, u, pi, tau_x, x0=None, eps=0.0):
    """Unroll Eq. 1 over a trial.

    u : (T, Nu)
    pi : (M,) held constant for the trial
    Returns xs (T, Nx), ys (T, Ny) after each step.
    """
    Nx = params.J_xx.shape[0]
    if x0 is None:
        x0 = jnp.zeros((Nx,), dtype=params.J_xx.dtype)

    def f(x, u_t):
        x_n = nrnn_step(x, u_t, pi, params, tau_x, eps)
        y = decode(x_n, params.W_qx)
        return x_n, (x_n, y)

    _, (xs, ys) = jax.lax.scan(f, x0, u)
    return xs, ys


def arnn_step(h, o, params: ARNNParams, tau_h, eps=0.0):
    """One slow inter-trial A-RNN step (Eq. 2).

    h_{k+1} = (1-τ_h) h_k + τ_h ( W_hh tanh(h_k) + W_oh o_k + b_h + ε_h )
    """
    drive = params.W_hh @ jnp.tanh(h) + params.W_oh @ o + params.b_h + eps
    return (1.0 - tau_h) * h + tau_h * drive


def actor_logits(h, params: ARNNParams):
    """W2 ReLU(W1 h + b1) + b2 (inside Eq. 3)."""
    return params.W2 @ jax.nn.relu(params.W1 @ h + params.b1) + params.b2


def value(h, params: ARNNParams):
    """Linear critic V(h) used by A2C."""
    return jnp.dot(params.W_v, h) + params.b_v


def context_from_h(h, params: ARNNParams):
    """Discrete context via argmax of softmax (Eqs. 3-4).

    Returns g (int), pi one-hot (M,), logits (M,), probs (M,).
    """
    logits = actor_logits(h, params)
    probs = jax.nn.softmax(logits)
    g = jnp.argmax(probs)
    M = logits.shape[-1]
    pi = jax.nn.one_hot(g, M, dtype=logits.dtype)
    return g, pi, logits, probs


def arnn_update(h, o, params: ARNNParams, tau_h, eps=0.0):
    """A-RNN step then emit one-hot context π (Eqs. 2-4)."""
    h_new = arnn_step(h, o, params, tau_h, eps)
    g, pi, logits, probs = context_from_h(h_new, params)
    return h_new, g, pi, logits, probs


def observation(pi, r):
    """o_k = (π_k, r_k) concatenated for the A-RNN."""
    r = jnp.asarray(r, dtype=pi.dtype).reshape(())
    return jnp.concatenate([pi, r[None]])


def init_nrnn(key, Nx=16, M=2, Nu=3, Ny=2, h_std=0.01):
    """N-RNN weights. J_xx, B_dx, W_qx as Table I; H_px scale is `h_std`."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    J_xx = jax.random.normal(k1, (Nx, Nx)) / jnp.sqrt(Nx)
    H_px = jax.random.normal(k2, (Nx, M)) * h_std
    B_dx = jax.random.normal(k3, (Nx, Nu)) / jnp.sqrt(Nu)
    W_qx = jax.random.normal(k4, (Ny, Nx)) / jnp.sqrt(Nx)
    return NRNNParams(J_xx=J_xx, H_px=H_px, B_dx=B_dx, W_qx=W_qx)


def init_arnn(key, Nh=8, M=2, No=None, hidden=None, w_hh_std=0.001, w_std=0.01):
    """A-RNN and actor/critic heads. Table I uses N(0, 0.001) for W_hh."""
    if No is None:
        No = M + 1
    if hidden is None:
        hidden = Nh
    keys = jax.random.split(key, 6)
    W_hh = jax.random.normal(keys[0], (Nh, Nh)) * w_hh_std
    W_oh = jax.random.normal(keys[1], (Nh, No)) / jnp.sqrt(No)
    b_h = jnp.zeros((Nh,))
    W1 = jax.random.normal(keys[2], (hidden, Nh)) * w_std
    b1 = jnp.zeros((hidden,))
    W2 = jax.random.normal(keys[3], (M, hidden)) * w_std
    b2 = jnp.zeros((M,))
    W_v = jax.random.normal(keys[4], (Nh,)) * w_std
    b_v = jnp.array(0.0)
    return ARNNParams(
        W_hh=W_hh,
        W_oh=W_oh,
        b_h=b_h,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        W_v=W_v,
        b_v=b_v,
    )
