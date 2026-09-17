"""Paper-faithful checks for the GMM toy, WTMetaD, Tweedie, and a tiny run."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_metalica.bias import cv_grid, deposit_hills, evaluate_V
from jax_metalica.gmm import log_p_t, responsibilities, score, toy_gmm, xhat0
from jax_metalica.kernels import grad_reward, pretrained_backward_mean, reward, tilted_backward_mean
from jax_metalica.metalica import MetalicaConfig, run
from jax_metalica.replica import replica_step, init_chain
from jax_metalica.sde import time_grid


def test_gmm_score_matches_central_difference():
    gmm = toy_gmm(0.0)
    xs = jnp.array([-3.0, -1.0, 0.0, 1.5, 3.0])
    ts = jnp.array([0.05, 0.3, 0.6])
    eps = 1e-3
    for t in ts:
        for x in xs:
            s = score(x, t, gmm)
            fd = (log_p_t(x + eps, t, gmm) - log_p_t(x - eps, t, gmm)) / (2.0 * eps)
            denom = jnp.maximum(jnp.abs(fd), 1e-3)
            rel = float(jnp.abs(s - fd) / denom)
            assert rel < 5e-3, f"rel={rel} at x={float(x)} t={float(t)}"


def test_tweedie_t0_identity_and_t05_finite():
    gmm = toy_gmm(0.0)
    x = jnp.asarray(1.7)
    x0_near = xhat0(x, 1e-4, gmm)
    np.testing.assert_allclose(float(x0_near), float(x), atol=5e-2)
    x0_mid = xhat0(x, 0.5, gmm)
    assert np.isfinite(float(x0_mid))


def test_responsibilities_sum_to_one():
    gmm = toy_gmm(0.0)
    q = responsibilities(jnp.asarray(0.3), 0.4, gmm)
    np.testing.assert_allclose(float(q.sum()), 1.0, atol=1e-6)
    assert q.shape == (2,)


def test_wtmetad_deposit_raises_local_hill():
    grid = cv_grid(81)
    V0 = grid.V
    V1, b = deposit_hills(grid.omega, V0, jnp.array([3.0]), h=0.1, sigma=0.2, gamma=2.0)
    assert jnp.all(b >= 0.0)
    assert float(V1.mean()) >= float(V0.mean())
    v_pos = float(evaluate_V(grid.omega, V1, 3.0))
    v_neg = float(evaluate_V(grid.omega, V1, -3.0))
    assert v_pos > v_neg


def test_V0_unbiased_means_muQ_equals_muP():
    gmm = toy_gmm(0.0)
    grid = cv_grid(41)
    y = jnp.asarray(-0.8)
    t_hi = jnp.asarray(0.35)
    dt = jnp.asarray(0.04)
    mu_p = pretrained_backward_mean(y, t_hi, dt, gmm)
    mu_q = tilted_backward_mean(y, t_hi, dt, grid.omega, grid.V, gmm, rho=0.5)
    np.testing.assert_allclose(mu_q, mu_p, atol=1e-6)
    np.testing.assert_allclose(float(reward(y, t_hi, grid.omega, grid.V, gmm)), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(grad_reward(y, t_hi, grid.omega, grid.V, gmm)), 0.0, atol=1e-6)


def test_tiny_metalica_deltaF_finite():
    key = jax.random.PRNGKey(2)
    cfg = MetalicaConfig(M=8, N=8, K=3, n_grid=41, eta=0.0)
    result = run(key, cfg)
    assert result.samples.shape == (32,)
    assert result.weights.shape == (32,)
    assert jnp.all(result.weights >= 0.0)
    np.testing.assert_allclose(float(result.weights.sum()), 1.0, atol=1e-5)
    assert np.isfinite(float(result.delta_F))


def test_jit_replica_step_and_deposit():
    key = jax.random.PRNGKey(3)
    gmm = toy_gmm(0.0)
    times = time_grid(8)
    grid = cv_grid(41)
    chain = init_chain(key, times, gmm)
    step = jax.jit(
        lambda k, ch, V: replica_step(k, ch, times, grid.omega, V, gmm, 0.5, 0)
    )
    dep = jax.jit(
        lambda V, xs: deposit_hills(grid.omega, V, xs, h=0.1, sigma=0.2, gamma=2.0)
    )
    ch2 = step(jax.random.PRNGKey(4), chain, grid.V)
    V2, b = dep(grid.V, ch2[:1])
    assert ch2.shape == chain.shape
    assert V2.shape == grid.V.shape
    assert jnp.all(b >= 0.0)
