"""Calibration recovers planted observation / process statistics (eqs 77–80)."""

import jax
import jax.numpy as jnp
import numpy as np

from quake.calibrate import observation_params, process_params


def test_observation_params_recover_planted():
    key = jax.random.PRNGKey(1)
    n, c = 80000, 4
    f = jax.random.normal(key, (n, c))
    gamma_true = jnp.array([0.8, 1.1, 0.55, 0.95])
    xi_true = jnp.array([0.2, -0.1, 0.0, 0.3])
    R_true = jnp.array([0.04, 0.01, 0.09, 0.025])
    noise = jax.random.normal(jax.random.fold_in(key, 1), (n, c)) * jnp.sqrt(R_true)
    f_hat = gamma_true * f + xi_true + noise

    gamma, xi, R = observation_params(f, f_hat, axis=0)
    np.testing.assert_allclose(np.asarray(gamma), np.asarray(gamma_true), rtol=0.03, atol=0.01)
    np.testing.assert_allclose(np.asarray(xi), np.asarray(xi_true), rtol=0.03, atol=0.01)
    np.testing.assert_allclose(np.asarray(R), np.asarray(R_true), rtol=0.05, atol=0.005)


def test_process_params_recover_planted():
    key = jax.random.PRNGKey(2)
    n, c = 50000, 3
    mu_true = jnp.array([0.1, -0.05, 0.2])
    Q_true = jnp.array([0.09, 0.04, 0.16])
    w = mu_true + jax.random.normal(key, (n, c)) * jnp.sqrt(Q_true)
    mu, Q = process_params(w, axis=0)
    np.testing.assert_allclose(np.asarray(mu), np.asarray(mu_true), rtol=0.04, atol=0.01)
    np.testing.assert_allclose(np.asarray(Q), np.asarray(Q_true), rtol=0.05, atol=0.005)
