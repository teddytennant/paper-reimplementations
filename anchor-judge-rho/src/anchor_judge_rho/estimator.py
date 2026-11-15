"""Closed-form identification of anchor–judge contamination ρ (Theorem 1)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from anchor_judge_rho.diagnostics import pair_denominator

DEN_ATOL = 1e-10


def _as_array(x: Any) -> jnp.ndarray:
    return jnp.asarray(x)


def mean_offdiag(cov: Any) -> jnp.ndarray:
    """Mean of off-diagonal entries of a square covariance matrix."""
    cov = _as_array(cov)
    p = cov.shape[0]
    mask = 1.0 - jnp.eye(p, dtype=cov.dtype)
    return (cov * mask).sum() / (p * (p - 1))


def moments_from_scores(judges: Any, anchors: Any) -> dict[str, Any]:
    """Second-moment summaries from an item panel.

    Parameters
    ----------
    judges : array (N, p)
    anchors : array (N, m)

    Both are mean-centered internally. ``K`` is the mean off-diagonal of the
    judge covariance; ``M[k]`` is the mean of Cov(J_j, A_k) over judges.
    """
    judges = _as_array(judges)
    anchors = _as_array(anchors)
    judges = judges - judges.mean(axis=0, keepdims=True)
    anchors = anchors - anchors.mean(axis=0, keepdims=True)
    n = judges.shape[0]
    p = judges.shape[1]
    m = anchors.shape[1]
    judge_cov = (judges.T @ judges) / (n - 1)
    anchor_cov = (anchors.T @ anchors) / (n - 1)
    cross = (judges.T @ anchors) / (n - 1)
    k = mean_offdiag(judge_cov)
    m_vec = cross.mean(axis=0)
    var_a = jnp.diag(anchor_cov)
    return {
        "K": k,
        "M": m_vec,
        "P": anchor_cov,
        "var_a": var_a,
        "judge_cov": judge_cov,
        "n": n,
        "p": p,
        "m": m,
    }


def pair_sigma_t2(
    k: Any,
    m_i: Any,
    m_j: Any,
    p_ij: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Theorem 1 for one anchor pair: (σ_t², denominator)."""
    den = pair_denominator(k, m_i, m_j, p_ij)
    num = _as_array(k) * _as_array(p_ij) - _as_array(m_i) * _as_array(m_j)
    return num / den, den


def _all_pair_sigma_t2(
    k: jnp.ndarray,
    m_vec: jnp.ndarray,
    p_mat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    m = m_vec.shape[0]
    i, j = jnp.triu_indices(m, k=1)
    p_ij = p_mat[i, j]
    den = (k + p_ij) - (m_vec[i] + m_vec[j])
    num = k * p_ij - m_vec[i] * m_vec[j]
    return num / den, den


def _average_sigma_t2(
    pair_st2: jnp.ndarray,
    pair_den: jnp.ndarray,
    den_atol: float = DEN_ATOL,
) -> jnp.ndarray:
    """Average pair σ_t², skipping pairs whose |denominator| is tiny."""
    valid = jnp.abs(pair_den) > den_atol
    n_valid = valid.sum()
    fallback = pair_st2.mean()
    weighted = jnp.where(valid, pair_st2, jnp.zeros_like(pair_st2)).sum()
    return jnp.where(n_valid > 0, weighted / n_valid, fallback)


def _rho_from_sigma_t2(
    k: jnp.ndarray,
    m_vec: jnp.ndarray,
    var_a: jnp.ndarray,
    sigma_t2: jnp.ndarray,
) -> dict[str, Any]:
    sigma_c2 = k - sigma_t2
    beta = m_vec - sigma_t2
    sigma_a2 = var_a - sigma_t2
    rho = beta / (jnp.sqrt(sigma_a2) * jnp.sqrt(sigma_c2))
    out_of_range = (sigma_c2 <= 0) | (sigma_t2 <= 0) | (sigma_t2 >= k)
    return {
        "sigma_t2": sigma_t2,
        "sigma_c2": sigma_c2,
        "beta": beta,
        "sigma_a2": sigma_a2,
        "rho": rho,
        "out_of_range": out_of_range,
        "K": k,
    }


def estimate(moments: dict[str, Any], den_atol: float = DEN_ATOL) -> dict[str, Any]:
    """Theorem 1 closed form, averaging σ_t² over well-identified anchor pairs.

    Returns σ_t², σ_c², β, σ_a², ρ, and the per-pair denominators. Heywood
    cases (σ_c² ≤ 0 or σ_t² ∉ (0, K)) are flagged in ``out_of_range`` and
    not clipped; |ρ̂| > 1 is allowed.
    """
    k = _as_array(moments["K"])
    m_vec = _as_array(moments["M"])
    p_mat = _as_array(moments["P"])
    var_a = _as_array(moments["var_a"])
    pair_st2, pair_den = _all_pair_sigma_t2(k, m_vec, p_mat)
    sigma_t2 = _average_sigma_t2(pair_st2, pair_den, den_atol=den_atol)
    out = _rho_from_sigma_t2(k, m_vec, var_a, sigma_t2)
    out["denominator"] = pair_den
    out["pair_sigma_t2"] = pair_st2
    out["pair_den"] = pair_den
    return out


def estimate_from_scores(judges: Any, anchors: Any, den_atol: float = DEN_ATOL) -> dict[str, Any]:
    """``moments_from_scores`` followed by ``estimate``."""
    return estimate(moments_from_scores(judges, anchors), den_atol=den_atol)


def estimate_clean_anchor(
    moments: dict[str, Any],
    trusted_k: int = 0,
) -> dict[str, Any]:
    """CT-C(M−1)-style estimator that forces the trusted anchor to be clean.

    Sets β_{trusted} = 0 so σ_t² = M_{trusted}. A contaminated trusted
    anchor then reports a contaminated companion as (near) clean.
    """
    k = _as_array(moments["K"])
    m_vec = _as_array(moments["M"])
    var_a = _as_array(moments["var_a"])
    sigma_t2 = m_vec[trusted_k]
    out = _rho_from_sigma_t2(k, m_vec, var_a, sigma_t2)
    out["rho"] = out["rho"].at[trusted_k].set(0.0)
    out["beta"] = out["beta"].at[trusted_k].set(0.0)
    out["trusted_k"] = trusted_k
    out["denominator"] = jnp.array([])
    out["pair_sigma_t2"] = jnp.array([sigma_t2])
    return out


def k_within_cross(judges: Any, families: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mean within-family and cross-family off-diagonal judge covariances."""
    judges = _as_array(judges)
    judges = judges - judges.mean(axis=0, keepdims=True)
    n, p = judges.shape
    cov = (judges.T @ judges) / (n - 1)
    fam = _as_array(families)
    same = fam[:, None] == fam[None, :]
    off = 1.0 - jnp.eye(p, dtype=cov.dtype)
    within = same.astype(cov.dtype) * off
    cross = (1.0 - same.astype(cov.dtype)) * off
    k_within = (cov * within).sum() / jnp.maximum(within.sum(), 1.0)
    k_cross = (cov * cross).sum() / jnp.maximum(cross.sum(), 1.0)
    return k_within, k_cross


def estimate_family_blocked(
    judges: Any,
    anchors: Any,
    families: Any,
    den_atol: float = DEN_ATOL,
) -> dict[str, Any]:
    """Theorem 1 with K computed from cross-family judge pairs only.

    Under J_j = t + c + f_{b(j)} + e_j, cross-family covariances equal
    σ_t² + σ_c², so the closed form is restored. ``K_within - K_cross``
    estimates σ_f² (Test C).
    """
    moments = moments_from_scores(judges, anchors)
    k_within, k_cross = k_within_cross(judges, families)
    moments = dict(moments)
    moments["K"] = k_cross
    moments["K_within"] = k_within
    moments["K_cross"] = k_cross
    out = estimate(moments, den_atol=den_atol)
    out["K_within"] = k_within
    out["K_cross"] = k_cross
    out["sigma_f2"] = k_within - k_cross
    return out
