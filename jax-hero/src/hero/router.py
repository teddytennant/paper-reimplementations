"""Shared HeRo router: state, linear-attention memory, and gate (eqs. 4-16)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from hero.config import HeroConfig
from hero.ops import (
    cosine_similarity,
    linear,
    memory_read,
    memory_write,
    phi,
    rms_norm,
)

RouterParams = dict[str, Any]


def init_shared_router(key: jax.Array, cfg: HeroConfig) -> RouterParams:
    """Encoder, memory, and router heads shared across routed blocks."""
    keys = jax.random.split(key, 16)
    d_s = cfg.d_s
    scale = 0.1

    def n(k, shape):
        return scale * jax.random.normal(k, shape, dtype=jnp.float32)

    return {
        "rms_scale": jnp.ones((cfg.d_model,), dtype=jnp.float32),
        "W_hd": n(keys[0], (cfg.d_h, cfg.d_model)),
        "W_hu": n(keys[1], (cfg.d_h, cfg.d_h)),
        "W_path1": n(keys[2], (cfg.d_a, 6)),
        "W_path2": n(keys[3], (cfg.d_a, cfg.d_a)),
        "W_q": n(keys[4], (cfg.d_m, d_s)),
        "W_k": n(keys[5], (cfg.d_m, d_s)),
        "W_v": n(keys[6], (cfg.d_m, d_s)),
        "W_n": n(keys[7], (cfg.d_cmp, d_s)),
        "W_c": n(keys[8], (cfg.d_cmp, cfg.d_m)),
        "W_theta1": n(keys[9], (cfg.d_h, d_s + cfg.d_m + 1)),
        "W_theta2": n(keys[10], (1, cfg.d_h)),
        "b_theta": jnp.zeros((1,), dtype=jnp.float32),
        "W_psi1": n(keys[11], (cfg.d_h, cfg.d_model)),
        "W_psi2": n(keys[12], (1, cfg.d_h)),
        "b_psi": jnp.zeros((1,), dtype=jnp.float32),
    }


def encode_hidden(hbar: jax.Array, router: RouterParams, cfg: HeroConfig) -> jax.Array:
    """z^h = W_hu SiLU(W_hd RMSNorm(hbar)) in R^{d_h} (eq. 5)."""
    x = rms_norm(hbar, router["rms_scale"], cfg.ln_eps)
    return linear(jax.nn.silu(linear(x, router["W_hd"])), router["W_hu"])


def encode_path(x_path: jax.Array, router: RouterParams) -> jax.Array:
    """z^a = FFN_path(x_path) (eq. 9)."""
    return linear(jax.nn.silu(linear(x_path, router["W_path1"])), router["W_path2"])


def path_features(
    j: int,
    n_routed: int,
    hbar: jax.Array,
    hbar_prev: jax.Array | None,
    delta_prev: jax.Array | None,
    p_hist_prev: jax.Array,
    m_hist_prev: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Six scalar path features (eqs. 4, 6-8).

    x_path = [r, q_rem, d, gamma, p_hist_prev, m_hist_prev]
    r = j / L_R, q_rem = (L_R - j) / L_R
    delta_lj = hbar_lj - hbar_{lj-1} (0 at j=1)
    d = ||delta||_2
    gamma = 0 if j <= 2 else cosine(delta_lj, delta_{lj-1})
    """
    batch_shape = hbar.shape[:-1]
    r = jnp.full(batch_shape, j / n_routed, dtype=hbar.dtype)
    q_rem = jnp.full(batch_shape, (n_routed - j) / n_routed, dtype=hbar.dtype)
    if j == 1 or hbar_prev is None:
        delta = jnp.zeros_like(hbar)
    else:
        delta = hbar - hbar_prev
    dist = jnp.linalg.norm(delta, axis=-1)
    if j <= 2 or delta_prev is None:
        gamma = jnp.zeros(batch_shape, dtype=hbar.dtype)
    else:
        gamma = cosine_similarity(delta, delta_prev, eps)
    x_path = jnp.stack(
        [r, q_rem, dist, gamma, p_hist_prev, m_hist_prev], axis=-1
    )
    return x_path, delta


def history_logit(
    state: jax.Array,
    context: jax.Array,
    nu: jax.Array,
    router: RouterParams,
) -> jax.Array:
    """a = f_theta([s; c; nu]) (eq. 14)."""
    feat = jnp.concatenate([state, context, nu[..., None]], axis=-1)
    hidden = jax.nn.silu(linear(feat, router["W_theta1"]))
    return linear(hidden, router["W_theta2"])[..., 0] + router["b_theta"][0]


def local_logit(hbar: jax.Array, router: RouterParams) -> jax.Array:
    """b = f_psi(hbar) (eq. 16)."""
    hidden = jax.nn.silu(linear(hbar, router["W_psi1"]))
    return linear(hidden, router["W_psi2"])[..., 0] + router["b_psi"][0]


def route_block(
    hbar: jax.Array,
    j: int,
    n_routed: int,
    router: RouterParams,
    eta: jax.Array,
    s_prev: jax.Array,
    zeta_prev: jax.Array,
    hbar_prev: jax.Array | None,
    delta_prev: jax.Array | None,
    p_hist_prev: jax.Array,
    m_hist_prev: jax.Array,
    cfg: HeroConfig,
) -> dict[str, jax.Array]:
    """One routed-block router step: encode, read, gate, then write.

    Encoder, memory, and heads are shared. eta (hence rho) is block-specific.
    """
    x_path, delta = path_features(
        j,
        n_routed,
        hbar,
        hbar_prev,
        delta_prev,
        p_hist_prev,
        m_hist_prev,
        cfg.eps,
    )
    z_h = encode_hidden(hbar, router, cfg)
    z_a = encode_path(x_path, router)
    state = jnp.concatenate([z_h, z_a], axis=-1)

    q = linear(state, router["W_q"])
    k = linear(state, router["W_k"])
    v = linear(state, router["W_v"])
    phi_q = phi(q)
    phi_k = phi(k)

    context = memory_read(phi_q, s_prev, zeta_prev, cfg.eps)

    if j == 1:
        nu = jnp.zeros(hbar.shape[:-1], dtype=hbar.dtype)
    else:
        nu = 1.0 - cosine_similarity(
            linear(state, router["W_n"]),
            linear(context, router["W_c"]),
            cfg.eps,
        )

    a = history_logit(state, context, nu, router)
    p_hist = jax.nn.sigmoid(a / cfg.t_hist)
    m_hist = m_hist_prev * p_hist
    b = local_logit(hbar, router)
    gate = jax.nn.sigmoid(a + b)

    rho = jax.nn.sigmoid(eta)
    s_new, zeta_new = memory_write(s_prev, zeta_prev, phi_k, v, rho)

    return {
        "s": state,
        "c": context,
        "q": q,
        "k": k,
        "v": v,
        "phi_k": phi_k,
        "rho": rho,
        "nu": nu,
        "a": a,
        "b": b,
        "g": gate,
        "p_hist": p_hist,
        "m_hist": m_hist,
        "delta": delta,
        "S": s_new,
        "zeta": zeta_new,
        "x_path": x_path,
    }
