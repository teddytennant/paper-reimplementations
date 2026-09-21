"""Predict/update identities and covariance Monte Carlo (paper eqs 10–16)."""

import jax
import jax.numpy as jnp
import numpy as np

from quake.kalman import predict, update
from quake.transition import observation_row, shift_register


def test_exact_observation_gain_to_one():
    """R → 0, γ = 1, ξ = 0: current-slot posterior mean ≈ observation, K[0] → 1."""
    window = 2
    A, b, G = shift_register(jnp.array([0.7, 0.3]), 0.1, window)
    mean = jnp.array([0.4, -0.2])
    cov = jnp.array([[0.8, 0.1], [0.1, 0.5]])
    mu, Q = 0.05, 0.2
    m_pred, P_pred = predict(mean, cov, A, b, G, mu, Q)

    gamma, xi, R = 1.0, 0.0, 1e-12
    f_hat = jnp.asarray(1.7)
    H_row = observation_row(gamma, window)
    m_post, _P_post, K = update(m_pred, P_pred, f_hat, H_row, xi, R)

    np.testing.assert_allclose(np.asarray(K[0]), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(m_post[0]), np.asarray(f_hat), atol=1e-5)


def test_joseph_covariance_matches_monte_carlo():
    """P = (I − KH) P matches empirical error covariance of the linear-Gaussian model."""
    window = 2
    A, b, G = shift_register(jnp.array([0.6, 0.3]), 0.05, window)
    mu, Q = 0.02, 0.08
    gamma, xi, R = 0.9, 0.1, 0.05
    m0 = jnp.array([1.0, 0.5])
    P0 = jnp.array([[0.5, 0.12], [0.12, 0.4]])
    m_pred, P_pred = predict(m0, P0, A, b, G, mu, Q)

    N = 40000
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    L = jnp.linalg.cholesky(P_pred)
    z = jax.random.normal(k1, (N, window))
    s = m_pred + z @ L.T
    eps = jax.random.normal(k2, (N,)) * jnp.sqrt(R)
    H_row = observation_row(gamma, window)
    y = s @ H_row + xi + eps

    m_post, P_kf, K = update(m_pred, P_pred, y, H_row, xi, R)
    err = s - m_post
    err0 = err - err.mean(axis=0, keepdims=True)
    P_mc = (err0.T @ err0) / N

    S = H_row @ P_pred @ H_row + R
    P_eq23 = P_pred - jnp.outer(K, K) * S
    rel_forms = float(jnp.linalg.norm(P_kf - P_eq23) / jnp.linalg.norm(P_eq23))
    assert rel_forms < 1e-5

    rel = float(jnp.linalg.norm(P_mc - P_kf) / jnp.linalg.norm(P_kf))
    assert rel < 0.05, f"relative Frobenius error {rel}"
