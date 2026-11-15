"""Discrete fiberwise-OT interval allocation and piecewise-linear Φ."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

LAMBDA_DDPM = 220.0
LAMBDA_FM = 450.0


def solve_eta(L, r, lam, n_iter: int = 80):
    """Solve Σ_k L_k / sqrt(η + λ r_k) = 1 for η > -λ min_k r_k (C.11).

    As η → -λ min r from above, Σ Δt → +∞. As η → +∞, Σ Δt → 0.
    The root of Σ_k L_k / sqrt(η + λ r_k) − 1 = 0 is unique.
    """
    L_np = np.asarray(jnp.asarray(L), dtype=np.float64).reshape(-1)
    r_np = np.asarray(jnp.asarray(r), dtype=np.float64).reshape(-1)
    lam_f = float(np.asarray(jnp.asarray(lam)))
    if L_np.size == 0:
        return jnp.asarray(1.0)

    r_min = float(np.min(r_np))
    lo = -lam_f * r_min
    L_sum = float(np.sum(L_np))
    if not np.isfinite(L_sum) or L_sum == 0.0:
        return jnp.asarray(lo + 1.0)

    def mass(eta):
        denom = eta + lam_f * r_np
        denom = np.maximum(denom, 1e-30)
        return float(np.sum(L_np / np.sqrt(denom)))

    scale = 1.0 + abs(lo) + abs(lam_f)
    left = lo + 1e-12 * scale
    if left <= lo:
        left = np.nextafter(lo, np.inf)
    for _ in range(48):
        if mass(left) > 1.0:
            break
        nxt = 0.5 * (lo + left)
        if nxt <= lo or not np.isfinite(nxt):
            left = np.nextafter(lo, np.inf)
            break
        left = nxt

    right = lo + (abs(L_sum) + 1.0) ** 2 + 1.0
    for _ in range(48):
        if mass(right) < 1.0:
            break
        right = lo + 2.0 * (right - lo)

    for _ in range(n_iter):
        mid = 0.5 * (left + right)
        if mass(mid) > 1.0:
            left = mid
        else:
            right = mid
    return jnp.asarray(0.5 * (left + right))


def allocate_weights(L, r, lam):
    """Discrete KKT allocation (C.11): Δt*_k = L_k / sqrt(η + λ r_k), Σ Δt* = 1.

    η is chosen so the weights sum to 1, with η > -λ min_k r_k so the sqrt is
    real. Higher risk therefore receives *less* schedule time.

    If λ = 0 or all r = 0, recovers the kinetic baseline Δt_k = L_k / Σ_j L_j.
    """
    L = jnp.asarray(L)
    r = jnp.asarray(r)
    lam = jnp.asarray(lam)
    L_sum = jnp.sum(L)
    w_base = L / jnp.where(L_sum == 0.0, 1.0, L_sum)
    use_base = (lam == 0) | jnp.all(r == 0)
    eta = solve_eta(L, r, lam)
    w = L / jnp.sqrt(eta + lam * r)
    w_sum = jnp.sum(w)
    w = w / jnp.where(w_sum == 0.0, 1.0, w_sum)
    return jnp.where(use_base, w_base, w)


def discrete_action(L, w, r, lam):
    """J(w) = Σ_k (L_k² / w_k + λ w_k r_k)  (C.10)."""
    L = jnp.asarray(L)
    w = jnp.asarray(w)
    r = jnp.asarray(r)
    lam = jnp.asarray(lam)
    return jnp.sum(jnp.square(L) / w + lam * w * r)


def phi_from_weights(w):
    """Piecewise-linear Φ, Φ⁻¹ from w on a uniform grid of the reference coordinate.

    Returns (tau_edges, t_edges) of length K+1 for w of length K, with
    tau uniform on [0, 1] and t the cumulative schedule-time allocation.
    """
    w = jnp.asarray(w)
    K = w.shape[0]
    tau_edges = jnp.linspace(0.0, 1.0, K + 1)
    t_edges = jnp.concatenate([jnp.zeros((1,), dtype=w.dtype), jnp.cumsum(w)])
    return tau_edges, t_edges


def apply_phi(tau, tau_edges, t_edges):
    """Φ(τ): reference coordinate → schedule time."""
    return jnp.interp(jnp.asarray(tau), tau_edges, t_edges)


def apply_phi_inv(t, tau_edges, t_edges):
    """Φ⁻¹(t): schedule time → reference coordinate."""
    return jnp.interp(jnp.asarray(t), t_edges, tau_edges)
