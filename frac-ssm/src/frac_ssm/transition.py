"""Frac state transition: Algorithm 1, sequential loop, and parallel scan.

Selective update (21-23) / Alg 1:
  τ̃ = τ ⊘ λ^{1/α}
  ρ̃ = exp(-Δ ⊘ τ̃)
  β̃ = 1 - ρ̃
  c = softmax_m(-α log τ + g_read(u))
  b = softmax_m(-α log τ + g_write(u))
  κ = β̃ ⊙ b
  S = ρ̃ ⊙ S_prev + κ ⊙ u
  h = Σ_m c_m ⊙ S_m

Affine recurrence (24): S_t = ρ_t ⊙ S_{t-1} + (κ_t ⊙ u_t).
Scan composition: (ρ, b) ⊕ (ρ', b') = (ρ' ρ, ρ' b + b').
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .bank import effective_timescales, implemented_rho_beta


def gated_mode_logits(u, weight, gate_logit):
    """g(u) = σ(p) W u, Appendix D. u: (..., H, dh), W: (H, M, dh)."""
    raw = jnp.einsum("hmd,...hd->...hm", weight, u)
    return jax.nn.sigmoid(gate_logit) * raw


def read_write_weights(alpha, tau, g_read, g_write):
    """c, b = softmax_m(-α log τ + g(u)) over modes."""
    prior = -alpha[..., None] * jnp.log(tau)
    c = jax.nn.softmax(prior + g_read, axis=-1)
    b = jax.nn.softmax(prior + g_write, axis=-1)
    return c, b


def alg1_rho_beta_kappa(delta, alpha, lam, tau, g_read, g_write):
    """Algorithm 1 steps 1-4: τ̃, ρ̃, β̃=1-ρ̃, read/write, κ=β̃ ⊙ b."""
    tau_tilde = effective_timescales(tau, lam[..., None], alpha[..., None])
    rho, beta = implemented_rho_beta(delta[..., None], tau_tilde)
    c, b_w = read_write_weights(alpha, tau, g_read, g_write)
    kappa = beta * b_w
    return rho, beta, kappa, c, b_w


def frac_step(u, delta, alpha, lam, S_prev, tau, g_read, g_write):
    """One Algorithm 1 step.

    Args:
        u: (..., H, dh)
        delta, alpha, lam: (..., H)
        S_prev: (..., H, M, dh)
        tau: (M,)
        g_read, g_write: (..., H, M)
    Returns:
        h: (..., H, dh), S: (..., H, M, dh)
    """
    rho, _beta, kappa, c, _b = alg1_rho_beta_kappa(
        delta, alpha, lam, tau, g_read, g_write
    )
    S = rho[..., None] * S_prev + kappa[..., None] * u[..., None, :]
    h = jnp.sum(c[..., None] * S, axis=-2)
    return h, S


def affine_compose(left, right):
    """(ρ, b) ⊕ (ρ', b') = (ρ' ρ, ρ' b + b'). Left is earlier in time."""
    rho_l, b_l = left
    rho_r, b_r = right
    return rho_r * rho_l, rho_r * b_l + b_r


def sequential_scan(rho, b, s0):
    """Sequential S_t = ρ_t ⊙ S_{t-1} + b_t.

    rho: (B, T, H, M, 1), b: (B, T, H, M, dh), s0: (B, H, M, dh)
    Returns S: (B, T, H, M, dh)
    """

    def step(s, elems):
        r, bb = elems
        s = r * s + bb
        return s, s

    rho_t = jnp.swapaxes(rho, 0, 1)
    b_t = jnp.swapaxes(b, 0, 1)
    _, s_seq = jax.lax.scan(step, s0, (rho_t, b_t))
    return jnp.swapaxes(s_seq, 0, 1)


def parallel_scan(rho, b, s0):
    """Associative scan of the same affine recurrence (eq. 24)."""
    rho_pref, b_pref = jax.lax.associative_scan(affine_compose, (rho, b), axis=1)
    return rho_pref * s0[:, None] + b_pref


def transition_sequence(
    u,
    delta,
    alpha,
    lam,
    tau,
    g_read,
    g_write,
    s0=None,
    parallel=True,
):
    """Run Algorithm 1 over a sequence.

    Args:
        u: (B, T, H, dh)
        delta, alpha, lam: (B, T, H)
        tau: (M,)
        g_read, g_write: (B, T, H, M)
        s0: (B, H, M, dh) or None (zeros)
        parallel: associative scan if True, sequential loop otherwise
    Returns:
        h: (B, T, H, dh), S: (B, T, H, M, dh)
    """
    B, T, H, dh = u.shape
    M = tau.shape[0]
    if s0 is None:
        s0 = jnp.zeros((B, H, M, dh), dtype=u.dtype)
    rho, _beta, kappa, c, _b = alg1_rho_beta_kappa(
        delta, alpha, lam, tau, g_read, g_write
    )
    rho_e = rho[..., None]
    b_inj = kappa[..., None] * u[..., None, :]
    scan = parallel_scan if parallel else sequential_scan
    S = scan(rho_e, b_inj, s0)
    h = jnp.sum(c[..., None] * S, axis=-2)
    return h, S
