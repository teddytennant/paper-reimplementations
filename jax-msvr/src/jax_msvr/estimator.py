"""MSVR / SOX inner-function trackers (paper eqs. 3, 4, 5, 8)."""

from __future__ import annotations

import jax.numpy as jnp


def correction_gamma(m, B1, beta):
    """Custom MSVR correction coefficient (section 3.2, Lemma 2).

    gamma_t = (m - B1) / (B1 * (1 - beta_t)) + (1 - beta_t)

    This is not the STORM choice gamma = 1 - beta. The extra
    (m - B1) / (B1 * (1 - beta)) term cancels the unsampled-block
    cross term. Derived under beta <= 1/2; the formula is still
    evaluated for larger beta.
    """
    return (m - B1) / (B1 * (1.0 - beta)) + (1.0 - beta)


def msvr_update(u, g_now, g_prev, sampled_idx, beta, gamma):
    """MSVR two-point inner tracker (eq. 4).

    For i in B1:
      u_bar = (1 - beta) u_{t-1} + beta g(w_t; xi) + gamma (g(w_t; xi) - g(w_{t-1}; xi))
      u_t = u_bar
    For i not in B1:
      u_t = u_{t-1}

    Same xi is used at w_t and w_{t-1} (two-point probe of the sampled block).

    Args:
      u: (m, k) previous inner estimator
      g_now: (B1, k) g_i(w_t; xi) on sampled blocks
      g_prev: (B1, k) g_i(w_{t-1}; xi) on the same xi
      sampled_idx: (B1,) int block indices
      beta, gamma: scalars
    Returns:
      u_new: (m, k)
    """
    u_bar = (1.0 - beta) * u[sampled_idx] + beta * g_now + gamma * (g_now - g_prev)
    return u.at[sampled_idx].set(u_bar)


def msvr_sp_update(u, g_now, jac_now, dw, sampled_idx, beta, gamma):
    """Single-point MSVR (eq. 5).

    Replaces g(w_t; xi) - g(w_{t-1}; xi) with Jac_g(w_t; xi) @ (w_t - w_{t-1}).
    Only one oracle evaluation of g_i at w_t; the difference is linearized.

    Args:
      u: (m, k)
      g_now: (B1, k)
      jac_now: (B1, k, d) Jacobian of g_i at w_t with the same xi
      dw: (d,) equal to w_t - w_{t-1}
      sampled_idx: (B1,)
      beta, gamma: scalars
    """
    linear = jnp.einsum("bkd,d->bk", jac_now, dw)
    u_tilde = (1.0 - beta) * u[sampled_idx] + beta * g_now + gamma * linear
    return u.at[sampled_idx].set(u_tilde)


def sox_update(u, g_now, sampled_idx, beta):
    """SOX moving-average inner tracker (eq. 3). Equivalent to MSVR with gamma=0."""
    u_bar = (1.0 - beta) * u[sampled_idx] + beta * g_now
    return u.at[sampled_idx].set(u_bar)


def finite_sum_g_hat(g_now, g_snap_stoch, g_snap_exact):
    """SVRG-style inner oracle used by finite-sum MSVR (eq. 8).

    g_hat = g_i(w_t; xi) - g_i(w_tau; xi) + g_i(w_tau)
    Unbiased for g_i(w_t) whenever xi is a uniform data index.
    """
    return g_now - g_snap_stoch + g_snap_exact


def msvr_fs_update(u, g_now, g_prev, g_snap_stoch, g_snap_exact, sampled_idx, beta, gamma):
    """Finite-sum MSVR inner tracker (eq. 8).

    u_hat = (1 - beta) u_{t-1} + beta g_hat + gamma (g(w_t; xi) - g(w_{t-1}; xi))
    """
    g_hat = finite_sum_g_hat(g_now, g_snap_stoch, g_snap_exact)
    u_bar = (1.0 - beta) * u[sampled_idx] + beta * g_hat + gamma * (g_now - g_prev)
    return u.at[sampled_idx].set(u_bar)


def block_chain_mean(grad_f_blocks, jac_blocks):
    """Mean over sampled blocks of (nabla f_i)^T (nabla g_i).

    grad_f_blocks: (B1, k)
    jac_blocks: (B1, k, d)
    returns: (d,)
    """
    b1 = grad_f_blocks.shape[0]
    return jnp.einsum("bk,bkd->d", grad_f_blocks, jac_blocks) / b1
