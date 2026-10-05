"""Seeded checks for the direct statistic, bootstrap, correction, and generator."""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from tgg import (
    band_edges,
    ce_test,
    consistency_empowered_statistic,
    direct_statistic,
    direct_test,
    generate_linear_process,
    plugin_scores,
    sample_covariance,
    sample_iid,
    screen_correction,
    suggested_block_length,
    tridiagonal_precision,
    two_sample_test,
)
from tgg.core import empirical_bootstrap_quantile, local_inverse, overlapping_block_sums


def _sigma(p, kappa):
    return jnp.linalg.inv(tridiagonal_precision(p, kappa))


def test_sample_covariance_uses_one_over_n():
    X = jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=jnp.float64)
    got = sample_covariance(X)
    expected = (X.T @ X) / X.shape[0]
    assert jnp.allclose(got, expected)
    unbiased = (X.T @ X) / (X.shape[0] - 1)
    assert not jnp.allclose(got, unbiased)


def test_population_statistic_is_zero_on_the_true_band():
    p = 12
    n = 80
    edges = band_edges(p, 1)
    for kappa in (0.2, 0.4):
        Sigma = _sigma(p, kappa)
        statistic = direct_statistic(Sigma, edges, n)
        assert abs(statistic) < 1e-8


def test_singular_neighborhood_inverse_is_zero():
    Sigma = jnp.array([[1.0, 1.0], [1.0, 1.0]], dtype=jnp.float64)
    omega = local_inverse(Sigma, [(0, 1)])
    assert jnp.allclose(omega, 0.0)


def test_plugin_scores_are_centered():
    n, p, kappa = 40, 8, 0.2
    X = sample_iid(n, _sigma(p, kappa), seed=3)
    Sigma = sample_covariance(X)
    scores, _pairs, _omega = plugin_scores(X, Sigma, band_edges(p, 1))
    assert scores.shape[0] == n
    assert float(jnp.max(jnp.abs(jnp.mean(scores, axis=0)))) < 1e-8


def test_overlapping_blocks_and_quantile_match_the_definition():
    scores = jnp.arange(6.0, dtype=jnp.float64)[:, None]
    blocks = overlapping_block_sums(scores, 3)
    expected = jnp.array([[3.0], [6.0], [9.0], [12.0]], dtype=jnp.float64)
    assert jnp.allclose(blocks, expected)
    # m = 5, 1 - alpha = 0.8, so the 4th order statistic.
    q = empirical_bootstrap_quantile(jnp.array([5.0, 1.0, 4.0, 2.0, 3.0]), 0.2)
    assert q == 4.0


def test_null_does_not_always_reject():
    n, p = 80, 12
    edges = band_edges(p, 1)
    l = math.floor(n ** (1.0 / 3.0))
    rejects = 0
    for seed in range(8):
        X = sample_iid(n, _sigma(p, 0.2), seed)
        rejects += int(direct_test(X, edges, l=l, alpha=0.05).reject)
    assert rejects / 8 <= 0.5


def test_wrong_edge_set_has_larger_statistic():
    n, p, kappa = 80, 12, 0.4
    edges = band_edges(p, 1)
    true_stats = []
    wrong_stats = []
    for seed in range(8):
        X = sample_iid(n, _sigma(p, kappa), seed)
        Sigma = sample_covariance(X)
        true_stats.append(direct_statistic(Sigma, edges, n))
        wrong_stats.append(direct_statistic(Sigma, [], n))
    assert float(np.mean(wrong_stats)) > float(np.mean(true_stats))


def test_smoke_block_length_floor_n_to_one_third():
    n, p = 80, 12
    l = math.floor(n ** (1.0 / 3.0))
    assert l == suggested_block_length(n, "one_third")
    assert l == 4
    X = sample_iid(n, _sigma(p, 0.2), seed=1)
    result = direct_test(X, band_edges(p, 1), l=l, alpha=0.05)
    assert math.isfinite(result.statistic)
    assert math.isfinite(result.critical_value)
    assert isinstance(result.reject, bool)
    ce = ce_test(X, band_edges(p, 1), l=l, alpha=0.05)
    assert math.isfinite(ce.statistic)
    assert isinstance(ce.reject, bool)


def test_screen_amplifies_near_zero_on_support_only():
    omega = jnp.array([1.5, 1.0e-8, 0.0], dtype=jnp.float64)
    tau = jnp.array([0.05, 0.05, 10.0], dtype=jnp.float64)
    on_support = jnp.array([True, True, False])
    corrected, delta = screen_correction(omega, tau, on_support, C_n=2.5, delta_n=1.0)
    assert float(delta[0]) == 0.0
    assert float(delta[1]) == 2.5
    assert float(delta[2]) == 0.0
    assert float(corrected[1]) == float(omega[1] + 2.5)
    assert float(corrected[0]) == 1.5


def test_pump_inflates_corrected_statistic_on_a_false_edge():
    # Sample covariance is exactly I, so a hypothesized edge has local-inverse
    # entry 0 and the diagonal entry is 1. The screen must amplify only the zero.
    X = math.sqrt(2.0) * jnp.eye(2, dtype=jnp.float64)
    tau = 0.1 * jnp.ones((2, 2), dtype=jnp.float64)
    statistic, _omega_tilde, delta, _tau = consistency_empowered_statistic(
        X,
        [(0, 1)],
        l=1,
        delta_n=1.0,
        C_n=2.0,
        tau=tau,
    )
    assert float(delta[0, 0]) == 0.0
    assert float(delta[1, 1]) == 0.0
    assert float(delta[1, 0]) == 2.0
    assert float(delta[0, 1]) == 2.0
    assert math.isclose(statistic, math.sqrt(2.0) * 2.0, rel_tol=0.0, abs_tol=1e-12)
    direct = direct_statistic(sample_covariance(X), [(0, 1)], n=2)
    assert direct == 0.0
    assert statistic > direct


def test_two_sample_returns_bool():
    n, p = 48, 8
    Sigma = _sigma(p, 0.2)
    X1 = sample_iid(n, Sigma, seed=11)
    X2 = sample_iid(n, Sigma, seed=12)
    decision = two_sample_test(X1, X2, l=math.floor(n ** (1.0 / 3.0)), alpha=0.1)
    assert isinstance(decision, bool)


def test_long_memory_acf_decays_from_lag_1_to_lag_10():
    acf1 = []
    acf10 = []
    Sigma = jnp.eye(1, dtype=jnp.float64)
    for seed in range(4):
        series = np.asarray(generate_linear_process(128, Sigma, beta=0.9, seed=seed))[:, 0]
        centered = series - series.mean()
        energy = float(np.dot(centered, centered))
        acf1.append(float(np.dot(centered[1:], centered[:-1]) / energy))
        acf10.append(float(np.dot(centered[10:], centered[:-10]) / energy))
    assert float(np.mean(acf1)) > float(np.mean(acf10))
