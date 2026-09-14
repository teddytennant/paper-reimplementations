"""Closed-form estimator of anchor–judge error correlation ρ (arXiv:2609.08826)."""

from anchor_judge_rho.diagnostics import (
    corollary1_denominator,
    pair_denominator,
    percentile_ci,
    bootstrap_rho,
    test_a,
    test_b,
    test_c,
    weak_id_T,
    weak_id_flag,
)
from anchor_judge_rho.estimator import (
    estimate,
    estimate_clean_anchor,
    estimate_family_blocked,
    estimate_from_scores,
    k_within_cross,
    moments_from_scores,
    pair_sigma_t2,
)
from anchor_judge_rho.simulate import (
    betas_from_rho,
    population_moments,
    simulate_panel,
)

__all__ = [
    "betas_from_rho",
    "bootstrap_rho",
    "corollary1_denominator",
    "estimate",
    "estimate_clean_anchor",
    "estimate_family_blocked",
    "estimate_from_scores",
    "k_within_cross",
    "moments_from_scores",
    "pair_denominator",
    "pair_sigma_t2",
    "percentile_ci",
    "population_moments",
    "simulate_panel",
    "test_a",
    "test_b",
    "test_c",
    "weak_id_T",
    "weak_id_flag",
]
