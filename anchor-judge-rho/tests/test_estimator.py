"""Equation-level tests for Theorem 1, Corollary 1, and related estimators."""

import jax
import jax.numpy as jnp
import pytest

from anchor_judge_rho import (
    betas_from_rho,
    corollary1_denominator,
    estimate,
    estimate_clean_anchor,
    estimate_family_blocked,
    estimate_from_scores,
    k_within_cross,
    pair_denominator,
    population_moments,
    simulate_panel,
    weak_id_flag,
)
from anchor_judge_rho.diagnostics import test_a as stat_a
from anchor_judge_rho.diagnostics import test_b as stat_b
from anchor_judge_rho.diagnostics import test_c as stat_c


SIGMA_T2 = 1.0
SIGMA_C2 = 0.8
SIGMA_A = 0.9
RHO = jnp.array([0.3, 0.7])
BETA = betas_from_rho(RHO, SIGMA_A, SIGMA_C2)


def _pop_two_anchor(**kwargs):
    params = dict(
        sigma_t2=SIGMA_T2,
        sigma_c2=SIGMA_C2,
        beta=BETA,
        sigma_a=SIGMA_A,
        p=4,
    )
    params.update(kwargs)
    return population_moments(**params)


def test_theorem1_recovers_population_moments():
    moments = _pop_two_anchor()
    est = estimate(moments)
    assert float(est["sigma_t2"]) == pytest.approx(SIGMA_T2, abs=1e-6)
    assert float(est["sigma_c2"]) == pytest.approx(SIGMA_C2, abs=1e-6)
    assert jnp.allclose(est["beta"], BETA, atol=1e-6)
    assert jnp.allclose(est["sigma_a2"], SIGMA_A**2, atol=1e-6)
    assert jnp.allclose(est["rho"], RHO, atol=1e-6)


def test_theorem1_three_anchors_pair_average():
    rho = jnp.array([0.2, 0.5, 0.8])
    beta = betas_from_rho(rho, SIGMA_A, SIGMA_C2)
    moments = population_moments(SIGMA_T2, SIGMA_C2, beta, SIGMA_A, p=4)
    est = estimate(moments)
    assert float(est["sigma_t2"]) == pytest.approx(SIGMA_T2, abs=1e-6)
    assert jnp.allclose(est["rho"], rho, atol=1e-6)
    assert est["pair_sigma_t2"].shape == (3,)
    assert float(stat_b(est["pair_sigma_t2"])) == pytest.approx(0.0, abs=1e-8)


def test_corollary1_denominator():
    moments = _pop_two_anchor()
    den = pair_denominator(moments["K"], moments["M"][0], moments["M"][1], moments["P"][0, 1])
    analytic = corollary1_denominator(SIGMA_C2, BETA[0], BETA[1])
    assert float(den) == pytest.approx(float(analytic), abs=1e-6)
    est = estimate(moments)
    assert float(est["denominator"][0]) == pytest.approx(float(analytic), abs=1e-6)


def test_finite_sample_rho():
    key = jax.random.PRNGKey(0)
    judges, anchors = simulate_panel(
        key,
        n=8000,
        p=4,
        m=2,
        sigma_t2=SIGMA_T2,
        sigma_c2=SIGMA_C2,
        rho=RHO,
        sigma_a=SIGMA_A,
    )
    est = estimate_from_scores(judges, anchors)
    assert jnp.max(jnp.abs(est["rho"] - RHO)) < 0.05


def test_clean_anchor_failure():
    true_rho2 = 0.5
    sigma_a = 0.9
    clean_hats = []
    closed_hats = []
    for rho1 in (0.0, 0.2, 0.4, 0.6):
        beta = betas_from_rho(jnp.array([rho1, true_rho2]), sigma_a, SIGMA_C2)
        moments = population_moments(SIGMA_T2, SIGMA_C2, beta, sigma_a, p=4)
        closed_hats.append(float(estimate(moments)["rho"][1]))
        trusted = estimate_clean_anchor(moments, trusted_k=0)
        clean_hats.append(float(trusted["rho"][1]))
        assert float(trusted["rho"][0]) == pytest.approx(0.0, abs=1e-12)

    # Two-anchor closed form stays at the true companion ρ for every ρ1.
    assert closed_hats == pytest.approx([true_rho2] * 4, abs=1e-5)
    # Clean-anchor is correct only while the trusted anchor is actually clean.
    assert clean_hats[0] == pytest.approx(true_rho2, abs=1e-5)
    # Then it degrades as the trusted anchor's contamination grows, reporting
    # the companion as far closer to clean than it is (paper Section 6).
    assert clean_hats[0] > clean_hats[1] > clean_hats[2] > clean_hats[3]
    assert abs(clean_hats[3] - true_rho2) > 0.4


def test_boundary_denominator_vanishes():
    beta1_star = SIGMA_C2  # Corollary 1: identification fails at β_k = σ_c²
    beta2 = float(BETA[1])
    beta = jnp.array([beta1_star, beta2])
    # σ_a must satisfy σ_a² ≥ β² / σ_c² so the implied ρ is feasible in moments.
    sigma_a = jnp.array([jnp.sqrt(beta1_star**2 / SIGMA_C2) + 0.1, SIGMA_A])
    moments = population_moments(SIGMA_T2, SIGMA_C2, beta, sigma_a, p=4)
    den = pair_denominator(moments["K"], moments["M"][0], moments["M"][1], moments["P"][0, 1])
    assert abs(float(den)) < 1e-8
    analytic = corollary1_denominator(SIGMA_C2, beta1_star, beta2)
    assert abs(float(analytic)) < 1e-12

    # Approach the boundary: |den| shrinks with |σ_c² − β1|.
    dens = []
    for gap in (0.2, 0.05, 0.01):
        b1 = SIGMA_C2 - gap
        b = jnp.array([b1, beta2])
        sa = jnp.array([jnp.sqrt(b1**2 / SIGMA_C2) + 0.1, SIGMA_A])
        mom = population_moments(SIGMA_T2, SIGMA_C2, b, sa, p=4)
        dens.append(
            abs(float(pair_denominator(mom["K"], mom["M"][0], mom["M"][1], mom["P"][0, 1])))
        )
    assert dens[0] > dens[1] > dens[2]


def test_family_blocked():
    key = jax.random.PRNGKey(7)
    families = jnp.array([0, 0, 1, 1, 2, 2])
    family_sigma = 0.7
    n = 8000
    judges, anchors = simulate_panel(
        key,
        n=n,
        p=6,
        m=2,
        sigma_t2=SIGMA_T2,
        sigma_c2=SIGMA_C2,
        rho=RHO,
        sigma_a=SIGMA_A,
        family_sigma=family_sigma,
        families=families,
    )
    naive = estimate_from_scores(judges, anchors)
    blocked = estimate_family_blocked(judges, anchors, families)
    true2 = float(RHO[1])
    naive2 = float(naive["rho"][1])
    blocked2 = float(blocked["rho"][1])
    assert abs(blocked2 - true2) < 0.05
    assert abs(naive2 - true2) > abs(blocked2 - true2)
    # Naive drifts toward a false "clean" (attenuation).
    assert naive2 < true2

    k_within, k_cross = k_within_cross(judges, families)
    sigma_f2_hat = float(stat_c(k_within, k_cross))
    assert sigma_f2_hat == pytest.approx(family_sigma**2, abs=0.08)
    assert float(blocked["sigma_f2"]) == pytest.approx(sigma_f2_hat, abs=1e-12)


def test_a_equal_vs_unequal_loadings():
    equal = population_moments(SIGMA_T2, SIGMA_C2, BETA, SIGMA_A, p=4)
    cv_eq = float(stat_a(equal["judge_cov"]))
    assert cv_eq == pytest.approx(0.0, abs=1e-12)

    g = jnp.array([0.5, 1.0, 1.5, 0.8])
    unequal = population_moments(
        SIGMA_T2, SIGMA_C2, BETA, SIGMA_A, p=4, judge_loadings=g
    )
    cv_un = float(stat_a(unequal["judge_cov"]))
    assert cv_un > 0.05
    assert cv_un > 10 * cv_eq + 0.05


def test_weak_id_helper_flags_small_t():
    assert bool(weak_id_flag(den=3.0, sd_boot_den=1.0, threshold=4.0))
    assert not bool(weak_id_flag(den=10.0, sd_boot_den=1.0, threshold=4.0))
