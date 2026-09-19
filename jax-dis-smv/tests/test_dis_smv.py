"""Equation-level tests for the disintegrated SMV kernel (arXiv:2609.16803)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln

from jax_dis_smv.bounds import (
    binary_kl,
    dis_r_certificate,
    dis_r_penalty,
    dis_v_certificate,
    dis_v_penalty,
    kl_inv,
)
from jax_dis_smv.dirichlet import (
    dirichlet_kl,
    dirichlet_renyi,
    log_beta,
    log_density_ratio,
    log_dirichlet_pdf,
)
from jax_dis_smv.majority_vote import empirical_risk_01, majority_vote_predict, vote_scores
from jax_dis_smv.train import algorithm_1, axis_aligned_stumps, split_data, synthetic_binary_2d


def test_majority_vote_three_voters_eq1_eq2():
    voter_preds = jnp.array(
        [
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=jnp.int32,
    )
    rho = jnp.array([0.5, 0.3, 0.2])
    scores = vote_scores(voter_preds, rho, n_classes=2)
    np.testing.assert_allclose(scores[0], [0.8, 0.2], atol=1e-6)
    np.testing.assert_allclose(scores[1], [0.3, 0.7], atol=1e-6)
    pred = majority_vote_predict(voter_preds, rho, n_classes=2)
    np.testing.assert_array_equal(np.asarray(pred), np.array([0, 1]))
    y = jnp.array([0, 1], dtype=jnp.int32)
    risk = empirical_risk_01(voter_preds, rho, y, n_classes=2)
    np.testing.assert_allclose(float(risk), 0.0, atol=1e-7)
    y_wrong = jnp.array([1, 1], dtype=jnp.int32)
    risk_wrong = empirical_risk_01(voter_preds, rho, y_wrong, n_classes=2)
    np.testing.assert_allclose(float(risk_wrong), 0.5, atol=1e-7)


def test_binary_kl_closed_form():
    p, q = 0.2, 0.5
    expected = p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    got = float(binary_kl(jnp.array(p), jnp.array(q)))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(float(binary_kl(0.0, 0.0)), 0.0, atol=1e-7)
    np.testing.assert_allclose(float(binary_kl(1.0, 1.0)), 0.0, atol=1e-7)
    np.testing.assert_allclose(float(binary_kl(0.3, 0.3)), 0.0, atol=1e-6)


def test_kl_inverse_recovers_epsilon():
    q = jnp.array(0.2)
    eps = jnp.array(0.1)
    p = kl_inv(q, eps)
    assert float(p) >= float(q) - 1e-6
    np.testing.assert_allclose(float(binary_kl(q, p)), float(eps), rtol=1e-4, atol=1e-5)
    p0 = kl_inv(q, jnp.array(0.0))
    np.testing.assert_allclose(float(p0), float(q), atol=1e-6)


def test_eq11_matches_log_pdf_ratio():
    cases = [
        (jnp.array([1.5, 2.0, 0.8]), jnp.array([0.5, 0.5, 0.5]), jnp.array([0.2, 0.3, 0.5])),
        (jnp.array([2.0, 2.0]), jnp.array([1.0, 3.0]), jnp.array([0.4, 0.6])),
        (jnp.array([0.7, 1.1, 1.4, 0.9]), jnp.array([1.0, 1.0, 1.0, 1.0]), jnp.array([0.1, 0.2, 0.3, 0.4])),
    ]
    for alpha, beta, rho in cases:
        rho = rho / jnp.sum(rho)
        got = log_density_ratio(rho, alpha, beta)
        pdf = log_dirichlet_pdf(rho, alpha) - log_dirichlet_pdf(rho, beta)
        explicit = (log_beta(beta) - log_beta(alpha)) + jnp.sum((alpha - beta) * jnp.log(rho))
        np.testing.assert_allclose(float(got), float(pdf), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(float(got), float(explicit), rtol=1e-5, atol=1e-6)


def test_eq12_identical_zero_and_kl_limit():
    alpha = jnp.array([1.2, 0.8, 2.0])
    d0 = dirichlet_renyi(alpha, alpha, 1.5)
    np.testing.assert_allclose(float(d0), 0.0, atol=1e-5)

    beta = jnp.array([0.9, 1.1, 1.4])
    d_near = dirichlet_renyi(alpha, beta, 1.001)
    kl = dirichlet_kl(alpha, beta)
    np.testing.assert_allclose(float(d_near), float(kl), rtol=5e-3, atol=5e-3)


def test_eq12_monte_carlo():
    alpha = jnp.array([1.2, 1.1, 0.9])
    beta = jnp.array([1.0, 1.0, 1.0])
    lam = 1.5
    closed = float(dirichlet_renyi(alpha, beta, lam))
    key = jax.random.PRNGKey(0)
    n_samples = 6000
    rho = jax.random.dirichlet(key, beta, (n_samples,))
    log_r = jax.vmap(lambda r: log_density_ratio(r, alpha, beta))(rho)
    log_mean = jax.nn.logsumexp(lam * log_r) - math.log(n_samples)
    mc = float(log_mean / (lam - 1.0))
    np.testing.assert_allclose(mc, closed, rtol=0.08, atol=0.05)


def test_eq14_dis_r_penalty_and_certificate():
    alpha = jnp.array([1.0, 2.0, 3.0])
    beta = jnp.array([0.5, 0.5, 0.5])
    rho = jnp.array([0.2, 0.3, 0.5])
    n = 100
    delta = 0.05
    rhat = jnp.array(0.1)
    got = float(dis_r_penalty(alpha, beta, rho, n, delta))
    logB = float(jnp.sum(gammaln(beta)) - gammaln(jnp.sum(beta))) - float(
        jnp.sum(gammaln(alpha)) - gammaln(jnp.sum(alpha))
    )
    expected = (
        logB + float(jnp.sum((alpha - beta) * jnp.log(rho))) + math.log(2.0 * math.sqrt(n) / delta)
    ) / n
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)
    cert = float(dis_r_certificate(rhat, alpha, beta, rho, n, delta))
    assert cert + 1e-6 >= float(rhat)
    np.testing.assert_allclose(float(binary_kl(rhat, cert)), got, rtol=1e-4, atol=1e-5)


def test_eq15_dis_v_penalty_and_certificate():
    alpha = jnp.array([1.0, 2.0, 3.0])
    beta = jnp.array([0.5, 0.5, 0.5])
    n = 100
    delta = 0.05
    lam = 1.5
    rhat = jnp.array(0.1)
    got = float(dis_v_penalty(alpha, beta, n, delta, lam))
    logB = float(log_beta(beta) - log_beta(alpha))
    mix = lam * alpha + (1.0 - lam) * beta
    logB_mix = float(log_beta(mix) - log_beta(alpha))
    expected = (
        ((2.0 * lam - 1.0) / (lam - 1.0)) * math.log(2.0 / delta)
        + logB
        + (1.0 / (lam - 1.0)) * logB_mix
        + math.log(2.0 * math.sqrt(n))
    ) / n
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)
    cert = float(dis_v_certificate(rhat, alpha, beta, n, delta, lam))
    assert cert + 1e-6 >= float(rhat)
    np.testing.assert_allclose(float(binary_kl(rhat, cert)), got, rtol=1e-4, atol=1e-5)


def test_algorithm_1_separable_stumps():
    key = jax.random.PRNGKey(7)
    k_data, k_train = jax.random.split(key)
    X, y = synthetic_binary_2d(k_data, n=48)
    voter_preds = axis_aligned_stumps(X, thresholds_per_dim=3)
    m = int(voter_preds.shape[1])
    beta = jnp.full((m,), 0.5)
    result = algorithm_1(
        k_train,
        voter_preds,
        y,
        beta,
        delta=0.05,
        n_epochs=30,
        eta=0.25,
        batch_size=None,
        objective="dis-r",
        slope=100.0,
    )
    assert jnp.all(result.alpha > 0.0)
    assert math.isfinite(float(result.bound))
    mean_rho = result.alpha / jnp.sum(result.alpha)
    mean_err = float(empirical_risk_01(voter_preds, mean_rho, y))
    keys = jax.random.split(jax.random.PRNGKey(123), 8)
    sample_err = [
        float(
            empirical_risk_01(
                voter_preds, jax.random.dirichlet(k, result.alpha), y
            )
        )
        for k in keys
    ]
    assert mean_err < 0.5
    assert sum(sample_err) / len(sample_err) < 0.5
    assert float(result.bound) + 1e-5 >= float(result.rhat)


def test_split_data_disjoint_covers():
    n = 10
    X = jnp.arange(n).reshape(n, 1).astype(jnp.float32)
    y = jnp.arange(n)
    (X1, y1), (X2, y2) = split_data(X, y, 4)
    assert int(y1.shape[0]) + int(y2.shape[0]) == n
    assert int(y1.shape[0]) == 4
    s1 = set(int(v) for v in np.asarray(y1).tolist())
    s2 = set(int(v) for v in np.asarray(y2).tolist())
    assert s1.isdisjoint(s2)
    assert s1 | s2 == set(range(n))
    key = jax.random.PRNGKey(1)
    (X1s, y1s), (X2s, y2s) = split_data(X, y, 6, key=key)
    assert int(y1s.shape[0]) + int(y2s.shape[0]) == n
    t1 = set(int(v) for v in np.asarray(y1s).tolist())
    t2 = set(int(v) for v in np.asarray(y2s).tolist())
    assert t1.isdisjoint(t2)
    assert t1 | t2 == set(range(n))
