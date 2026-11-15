"""Population moments and Gaussian panels under Assumption 1."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from anchor_judge_rho.estimator import mean_offdiag


def _broadcast_sigma_a(sigma_a: Any, m: int) -> jnp.ndarray:
    sigma_a = jnp.asarray(sigma_a, dtype=float)
    return jnp.broadcast_to(jnp.atleast_1d(sigma_a), (m,))


def betas_from_rho(rho: Any, sigma_a: Any, sigma_c2: Any) -> jnp.ndarray:
    """β_k = ρ_k σ_{a k} σ_c."""
    rho = jnp.atleast_1d(jnp.asarray(rho, dtype=float))
    sigma_a = _broadcast_sigma_a(sigma_a, int(rho.shape[0]))
    return rho * sigma_a * jnp.sqrt(jnp.asarray(sigma_c2, dtype=float))


def population_moments(
    sigma_t2: Any,
    sigma_c2: Any,
    beta: Any,
    sigma_a: Any,
    p: int = 4,
    sigma_e: float = 1.0,
    judge_loadings: Any | None = None,
) -> dict[str, Any]:
    """Exact second moments of the single-common-factor model."""
    sigma_t2 = jnp.asarray(sigma_t2, dtype=float)
    sigma_c2 = jnp.asarray(sigma_c2, dtype=float)
    beta = jnp.atleast_1d(jnp.asarray(beta, dtype=float))
    m = int(beta.shape[0])
    sigma_a = _broadcast_sigma_a(sigma_a, m)
    var_a = sigma_t2 + sigma_a**2
    p_mat = sigma_t2 + jnp.outer(beta, beta) / sigma_c2
    p_mat = p_mat.at[jnp.diag_indices(m)].set(var_a)
    if judge_loadings is None:
        g = jnp.ones((p,), dtype=float)
    else:
        g = jnp.asarray(judge_loadings, dtype=float)
    judge_cov = sigma_t2 + sigma_c2 * jnp.outer(g, g)
    judge_cov = judge_cov.at[jnp.diag_indices(p)].add(sigma_e**2)
    k = mean_offdiag(judge_cov)
    m_vec = sigma_t2 + beta * g.mean()
    # Unit loadings: Cov(J_j, A_k) = σ_t² + β_k for every j, so M_k = σ_t² + β_k.
    # With heterogeneous g_j, Cov(J_j, A_k) = σ_t² + g_j β_k, mean over j is above.
    if judge_loadings is None:
        m_vec = sigma_t2 + beta
        k = sigma_t2 + sigma_c2
    return {
        "K": k,
        "M": m_vec,
        "P": p_mat,
        "var_a": var_a,
        "judge_cov": judge_cov,
        "p": p,
        "m": m,
    }


def simulate_panel(
    key: Any,
    n: int,
    p: int,
    m: int | None = None,
    sigma_t2: float = 1.0,
    sigma_c2: float = 0.8,
    rho: Any = (0.3, 0.7),
    sigma_a: Any = 0.9,
    sigma_e: float = 1.0,
    family_sigma: float = 0.0,
    families: Any | None = None,
    judge_loadings: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Draw a mean-zero Gaussian panel.

    J_j = t + g_j c + f_{b(j)} + e_j,
    A_k = t + (β_k / σ_c²) c + u_k,
    with β_k = ρ_k σ_{a k} σ_c and Var(u_k) = σ_{a k}² (1 − ρ_k²).
    """
    rho = jnp.atleast_1d(jnp.asarray(rho, dtype=float))
    if m is None:
        m = int(rho.shape[0])
    sigma_a = _broadcast_sigma_a(sigma_a, m)
    beta = betas_from_rho(rho, sigma_a, sigma_c2)
    if judge_loadings is None:
        g = jnp.ones((p,), dtype=float)
    else:
        g = jnp.asarray(judge_loadings, dtype=float)

    keys = jax.random.split(key, 5)
    t = jax.random.normal(keys[0], (n,)) * jnp.sqrt(sigma_t2)
    c = jax.random.normal(keys[1], (n,)) * jnp.sqrt(sigma_c2)
    e = jax.random.normal(keys[2], (n, p)) * sigma_e
    var_u = sigma_a**2 * (1.0 - rho**2)
    u = jax.random.normal(keys[3], (n, m)) * jnp.sqrt(jnp.maximum(var_u, 0.0))

    judges = t[:, None] + g[None, :] * c[:, None] + e
    if family_sigma and families is not None:
        fam = jnp.asarray(families)
        n_fam = int(jnp.max(fam)) + 1
        f = jax.random.normal(keys[4], (n, n_fam)) * family_sigma
        judges = judges + f[:, fam]

    lam = beta / sigma_c2
    anchors = t[:, None] + lam[None, :] * c[:, None] + u
    return judges, anchors
