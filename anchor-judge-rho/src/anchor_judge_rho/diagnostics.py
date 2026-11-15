"""Diagnostic statistics (Test A/B/C and the weak-identification screen).

Thresholds are not the paper's 1000-seed 95th percentiles; callers compare
the raw statistics (or apply their own cutoffs).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def pair_denominator(k: Any, m1: Any, m2: Any, p12: Any) -> jnp.ndarray:
    """Theorem 1 denominator (K + P12) − (M1 + M2)."""
    return (jnp.asarray(k) + jnp.asarray(p12)) - (jnp.asarray(m1) + jnp.asarray(m2))


def corollary1_denominator(sigma_c2: Any, beta1: Any, beta2: Any) -> jnp.ndarray:
    """Analytic Corollary 1: (σ_c² − β1)(σ_c² − β2) / σ_c²."""
    sigma_c2 = jnp.asarray(sigma_c2)
    return (sigma_c2 - jnp.asarray(beta1)) * (sigma_c2 - jnp.asarray(beta2)) / sigma_c2


def test_a(judge_cov: Any) -> jnp.ndarray:
    """Coefficient of variation of unique off-diagonal judge covariances.

    Needs p ≥ 3. Under Assumption 1 the off-diagonals are equal, so CV ≈ 0
    on exact moments.
    """
    cov = jnp.asarray(judge_cov)
    p = cov.shape[0]
    i, j = jnp.triu_indices(p, k=1)
    vals = cov[i, j]
    mean = jnp.mean(vals)
    sd = jnp.std(vals, ddof=0)
    return sd / jnp.abs(mean)


def test_b(pair_sigma_t2: Any) -> jnp.ndarray:
    """Sample sd of pair-wise σ_t² estimates (needs m ≥ 3)."""
    x = jnp.asarray(pair_sigma_t2).reshape(-1)
    return jnp.std(x, ddof=1)


def test_c(k_within: Any, k_cross: Any) -> jnp.ndarray:
    """Family-block statistic K_within − K_cross (= σ_f² under the model)."""
    return jnp.asarray(k_within) - jnp.asarray(k_cross)


def weak_id_T(den: Any, sd_boot_den: Any) -> jnp.ndarray:
    """Studentized denominator T = |den| / SD_boot(den)."""
    return jnp.abs(jnp.asarray(den)) / jnp.asarray(sd_boot_den)


def weak_id_flag(den: Any, sd_boot_den: Any, threshold: float = 4.0) -> jnp.ndarray:
    """Flag weak identification when T < 4 (paper Section 6)."""
    return weak_id_T(den, sd_boot_den) < threshold


def bootstrap_rho(
    key: Any,
    judges: Any,
    anchors: Any,
    n_boot: int = 300,
    den_atol: float = 1e-10,
) -> jnp.ndarray:
    """Percentile bootstrap of ρ over items. Returns (n_boot, m)."""
    from anchor_judge_rho.estimator import estimate_from_scores

    judges = jnp.asarray(judges)
    anchors = jnp.asarray(anchors)
    n = judges.shape[0]

    def one(k):
        idx = jax.random.choice(k, n, shape=(n,), replace=True)
        return estimate_from_scores(judges[idx], anchors[idx], den_atol=den_atol)["rho"]

    keys = jax.random.split(key, n_boot)
    return jax.vmap(one)(keys)


def percentile_ci(samples: Any, level: float = 0.95) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Percentile interval from a bootstrap sample array (leading axis = draws)."""
    samples = jnp.asarray(samples)
    alpha = 1.0 - level
    lo = jnp.quantile(samples, alpha / 2.0, axis=0)
    hi = jnp.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return lo, hi
