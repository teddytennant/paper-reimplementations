"""Girsanov weights are the exact likelihood ratio (Eqs. 12, 76); stable, mean ~1 at tilt 0."""

import jax
import jax.numpy as jnp

from slisde.girsanov import (
    ess,
    log_dp_over_dq,
    log_dq_over_dp,
    snis_weights,
    tilt_innovation,
)
from slisde.sde import sample_increments


def test_zero_tilt_unit_weights():
    B, N, m, dt = 32, 10, 2, 0.1
    u = jnp.zeros((B, N, m))
    dW = jnp.ones((B, N, m))
    log_w = log_dp_over_dq(u, dW, dt, mu=jnp.zeros((m,)), z0=jnp.zeros((B, m)))
    assert jnp.all(jnp.isfinite(log_w))
    assert jnp.allclose(log_w, 0.0)
    w = jnp.exp(log_w)
    assert jnp.allclose(w.mean(), 1.0)
    nw = snis_weights(log_w)
    assert jnp.allclose(nw, 1.0 / B)


def test_exact_ratio_constant_drift_tilt():
    """1D Brownian with constant control u: log dP/dQ = -u W^Q_T - 1/2 u^2 T."""
    u_val, T, n = 0.7, 1.25, 20
    dt = T / n
    key = jax.random.PRNGKey(4)
    d_eps_Q = sample_increments(key, 8, n, 1, dt)
    u = jnp.full_like(d_eps_Q, u_val)
    log_w = log_dp_over_dq(u, d_eps_Q, dt)
    closed = -u_val * d_eps_Q.sum(axis=(1, 2)) - 0.5 * u_val**2 * T
    assert jnp.allclose(log_w, closed, atol=1e-6)
    assert jnp.all(jnp.isfinite(log_w))

    d_eps_P = tilt_innovation(d_eps_Q, u, dt)
    log_qp = log_dq_over_dp(u, d_eps_P, dt)
    # dQ/dP = 1 / (dP/dQ)
    assert jnp.allclose(log_qp, -log_w, atol=1e-5)


def test_mean_weight_near_one_under_q():
    """E_Q[dP/dQ] = 1 (Eq. 12 / Thm. 5.2)."""
    u_val, T, n, B = 0.35, 1.0, 16, 8192
    dt = T / n
    key = jax.random.PRNGKey(5)
    d_eps_Q = sample_increments(key, B, n, 1, dt)
    u = jnp.full_like(d_eps_Q, u_val)
    w = jnp.exp(log_dp_over_dq(u, d_eps_Q, dt))
    assert jnp.all(jnp.isfinite(jnp.log(w)))
    assert jnp.allclose(w.mean(), 1.0, atol=0.08)


def test_mean_weight_with_initial_shift():
    d, B = 3, 4096
    key = jax.random.PRNGKey(6)
    kz, ke = jax.random.split(key)
    mu = jnp.array([0.2, -0.1, 0.05])
    xi = jax.random.normal(kz, (B, d))
    z0 = mu + xi
    n, dt = 8, 0.05
    d_eps_Q = sample_increments(ke, B, n, d, dt)
    u = jnp.zeros_like(d_eps_Q)
    log_w = log_dp_over_dq(u, d_eps_Q, dt, mu=mu, z0=z0)
    w = jnp.exp(log_w)
    assert jnp.all(jnp.isfinite(log_w))
    assert jnp.allclose(w.mean(), 1.0, atol=0.1)


def test_snis_and_ess_finite():
    key = jax.random.PRNGKey(7)
    dW = sample_increments(key, 64, 12, 1, 0.1)
    u = 0.2 * jnp.ones_like(dW)
    log_w = log_dp_over_dq(u, dW, 0.1)
    nw = snis_weights(log_w)
    assert jnp.all(jnp.isfinite(nw))
    assert jnp.allclose(nw.sum(), 1.0)
    e = ess(log_w)
    assert jnp.isfinite(e)
    assert e > 1.0
