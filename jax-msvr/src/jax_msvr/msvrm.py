"""MSVRM-v1 / v2 / v3 parameter updates (paper eqs. 6, 7, 9)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_msvr.estimator import (
    block_chain_mean,
    msvr_fs_update,
    msvr_update,
)


class MSVRMState(NamedTuple):
    """Explicit pytree of MSVRM iterates.

    w: (d,) current parameters
    u: (m, k) inner estimator u_t after the last inner update, which is
       u_{t-1} at the start of the next step
    z: (d,) gradient tracker
    u_prev: (m, k) u_{t-2}, needed by v2/v3 STORM correction
    w_prev: (d,) previous parameters
    t: iteration counter (1-based, matching Algorithm 1/2)
    w_snap, u_snap_prev, g_snap, jac_snap: v3 snapshot fields
    """

    w: jnp.ndarray
    u: jnp.ndarray
    z: jnp.ndarray
    u_prev: jnp.ndarray
    w_prev: jnp.ndarray
    t: int
    w_snap: jnp.ndarray
    u_snap_prev: jnp.ndarray
    g_snap: jnp.ndarray
    jac_snap: jnp.ndarray


def init_msvrm_state(w, problem=None, m=None, k=None, u=None, z=None):
    """Initialize trackers. If `problem` is given, snapshot fields use exact g(w)."""
    w = jnp.asarray(w)
    d = int(w.shape[0])
    if problem is not None:
        m, k = problem.m, problem.k
        g_snap = problem.g_exact(w)
        jac_snap = problem.jac_exact(w)
        if u is None:
            u = g_snap
    else:
        if m is None or k is None:
            raise ValueError("pass problem or both m and k")
        g_snap = jnp.zeros((m, k), dtype=w.dtype)
        jac_snap = jnp.zeros((m, k, d), dtype=w.dtype)
        if u is None:
            u = jnp.zeros((m, k), dtype=w.dtype)
    if z is None:
        z = jnp.zeros((d,), dtype=w.dtype)
    u = jnp.asarray(u)
    z = jnp.asarray(z)
    return MSVRMState(
        w=w,
        u=u,
        z=z,
        u_prev=u,
        w_prev=w,
        t=1,
        w_snap=w,
        u_snap_prev=u,
        g_snap=g_snap,
        jac_snap=jac_snap,
    )


def apply_w_update(w, z, eta, adaptive=False, eps=1e-8):
    """w <- w - eta z, or AdaMSVRM w <- w - eta z / (||z|| + eps)."""
    if adaptive:
        return w - eta * z / (jnp.linalg.norm(z) + eps)
    return w - eta * z


def _sample_probe(oracle, w, w_prev, key, B1):
    k_blocks, k_probe = jax.random.split(key)
    idx = oracle.sample_blocks(k_blocks, B1)
    g_now, jac_now = oracle.probe(w, idx, k_probe)
    g_prev, jac_prev = oracle.probe(w_prev, idx, k_probe)
    return idx, g_now, g_prev, jac_now, jac_prev


def msvrm_v1_step(
    state,
    oracle,
    key,
    *,
    B1,
    alpha,
    beta,
    gamma,
    eta,
    adaptive=False,
    eps=1e-8,
):
    """One step of MSVRM-v1 (Algorithm 1 + eq. 6).

    Inner estimator: MSVR (eq. 4).
    Gradient tracker uses nabla f_i(u_{t-1}), not nabla f_i(u_t), so the
    chain-rule factor does not share randomness with the just-updated u_t.
    """
    idx, g_now, g_prev, jac_now, _jac_prev = _sample_probe(
        oracle, state.w, state.w_prev, key, B1
    )
    u_old = state.u
    u_new = msvr_update(u_old, g_now, g_prev, idx, beta, gamma)
    chain = block_chain_mean(oracle.grad_f(u_old)[idx], jac_now)
    z_new = (1.0 - alpha) * state.z + alpha * chain
    w_new = apply_w_update(state.w, z_new, eta, adaptive=adaptive, eps=eps)
    return state._replace(
        w=w_new,
        u=u_new,
        z=z_new,
        u_prev=u_old,
        w_prev=state.w,
        t=state.t + 1,
    )


def msvrm_v2_step(
    state,
    oracle,
    key,
    *,
    B1,
    alpha,
    beta,
    gamma,
    eta,
    adaptive=False,
    eps=1e-8,
):
    """One step of MSVRM-v2 (Algorithm 1 + eq. 7).

    STORM-like gradient tracker with the same xi at w_t and w_{t-1}.
    Needs u_{t-2} (state.u_prev).
    """
    idx, g_now, g_prev, jac_now, jac_prev = _sample_probe(
        oracle, state.w, state.w_prev, key, B1
    )
    u_old = state.u
    u_new = msvr_update(u_old, g_now, g_prev, idx, beta, gamma)
    chain_now = block_chain_mean(oracle.grad_f(u_old)[idx], jac_now)
    chain_prev = block_chain_mean(oracle.grad_f(state.u_prev)[idx], jac_prev)
    z_new = (1.0 - alpha) * state.z + chain_now - (1.0 - alpha) * chain_prev
    w_new = apply_w_update(state.w, z_new, eta, adaptive=adaptive, eps=eps)
    return state._replace(
        w=w_new,
        u=u_new,
        z=z_new,
        u_prev=u_old,
        w_prev=state.w,
        t=state.t + 1,
    )


def msvrm_v3_step(
    state,
    oracle,
    key,
    *,
    B1,
    alpha,
    beta,
    gamma,
    eta,
    I=10**9,
    adaptive=False,
    eps=1e-8,
):
    """One step of MSVRM-v3 (Algorithm 2 + eqs. 8-9).

    Every I steps take a snapshot w_tau and the exact g_i(w_tau) for all i.
    """
    if state.t % int(I) == 0:
        state = state._replace(
            w_snap=state.w,
            u_snap_prev=state.u,
            g_snap=oracle.g_exact(state.w),
            jac_snap=oracle.jac_exact(state.w),
        )
    k_blocks, k_probe = jax.random.split(key)
    idx = oracle.sample_blocks(k_blocks, B1)
    g_now, jac_now = oracle.probe(state.w, idx, k_probe)
    g_prev, jac_prev = oracle.probe(state.w_prev, idx, k_probe)
    g_snap_stoch, jac_snap_stoch = oracle.probe(state.w_snap, idx, k_probe)

    u_old = state.u
    u_new = msvr_fs_update(
        u_old,
        g_now,
        g_prev,
        g_snap_stoch,
        state.g_snap[idx],
        idx,
        beta,
        gamma,
    )

    full_snap = jnp.einsum(
        "mk,mkd->d", oracle.grad_f(state.u_snap_prev), state.jac_snap
    ) / u_old.shape[0]
    svrg = block_chain_mean(oracle.grad_f(u_old)[idx], jac_now) - block_chain_mean(
        oracle.grad_f(state.u_snap_prev)[idx], jac_snap_stoch
    )
    h = full_snap + svrg
    chain_now = block_chain_mean(oracle.grad_f(u_old)[idx], jac_now)
    chain_prev = block_chain_mean(oracle.grad_f(state.u_prev)[idx], jac_prev)
    z_new = (
        (1.0 - alpha) * state.z
        + alpha * h
        + (1.0 - alpha) * chain_now
        - (1.0 - alpha) * chain_prev
    )
    w_new = apply_w_update(state.w, z_new, eta, adaptive=adaptive, eps=eps)
    return state._replace(
        w=w_new,
        u=u_new,
        z=z_new,
        u_prev=u_old,
        w_prev=state.w,
        t=state.t + 1,
    )
