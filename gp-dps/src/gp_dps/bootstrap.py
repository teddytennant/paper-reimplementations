"""LOO residuals, multiplier bootstrap (eq 9), and studentized p-values (§4.4)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from gp_dps.gp import alpha_from_chol, kn_inv_diag


def loo_residuals(L: Array, y: Array) -> Array:
    """r_i = alpha_i / [Kn^{-1}]_{ii}  (Algorithm 1 step 3; RW 2006 §5.4.2)."""
    alpha = alpha_from_chol(L, y)
    inv_diag = kn_inv_diag(L)
    return alpha / inv_diag


def multiplier_bootstrap(A: Array, r: Array, key: Array, B: int) -> Array:
    """nu_j^{(b)} = n^{-1} ||A_j (r ⊙ e^{(b)})||^2 with e^{(b)} ~ N(0, I_n).

    A: (D, n, n), r: (n,). Returns (B, D).
    """
    n = r.shape[0]
    e = jax.random.normal(key, (B, n), dtype=r.dtype)
    re = e * r[None, :]
    # (D, n, n) @ (n, B) -> (D, n, B) then mean over n
    Aj_re = jnp.einsum("djs,sb->dbj", A, re.T)
    return jnp.mean(Aj_re ** 2, axis=-1).T


def studentized_pvalues(nu: Array, nu_boot: Array) -> tuple[Array, Array, Array, Array, Array]:
    """Studentize (§4.4): T_j = (nu_j - bbar_j)/s_j, same for bootstrap draws.

    p_j = (1 + count(T_j^{(b)} >= T_j)) / (B+1).
    Returns p, T, T_boot, bbar, s.
    """
    bbar = jnp.mean(nu_boot, axis=0)
    s = jnp.std(nu_boot, axis=0, ddof=1)
    s_safe = jnp.where(s > 0, s, jnp.array(1.0, dtype=nu.dtype))
    T = (nu - bbar) / s_safe
    T_boot = (nu_boot - bbar[None, :]) / s_safe[None, :]
    # If s==0, compare nu to bbar directly (degenerate studentization).
    T = jnp.where(s > 0, T, jnp.where(nu > bbar, jnp.inf, 0.0))
    T_boot = jnp.where(s[None, :] > 0, T_boot, jnp.where(nu_boot > bbar[None, :], jnp.inf, 0.0))
    ge = jnp.sum(T_boot >= T[None, :], axis=0)
    B = nu_boot.shape[0]
    p = (1.0 + ge) / (B + 1.0)
    return p, T, T_boot, bbar, s
